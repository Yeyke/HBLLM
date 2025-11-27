import time
import os
import torch
import torch.nn as nn

from bigptq import HBRAGPTQ
from haarbinary import HaarBinarization
from modelutils import find_layers


def get_model(model, model_path):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    if "opt" in model:
        from transformers import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model:
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        model.seqlen = 2048
    else:
        raise ValueError("Unsupported model type")
    return model


"""
The function is employed to calibrate and quantize models layer by layer.
"""


@torch.no_grad()
def quant_sequential(model, dataloader, dev):
    print("Starting ...")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False  

    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "llama" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp  
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0]) 
    for batch in dataloader:
        try:
            model(batch[0].to(dev))  
        except ValueError:
            pass 
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "llama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            if (
                not (args.minlayer <= i < args.maxlayer and args.quant_only in name)
            ) == (not args.invert):
                continue

            braq_quantizer = HaarBinarization(
                subset[name].weight,
                method=args.quant_method,
                groupsize=groupsize,
                group_partition=args.group_partition,
                share_mean=args.share_mean,

            )
            gptq[name] = HBRAGPTQ(
                subset[name],
                braq_quantizer,
                salient_metric=args.salient_metric,
                disable_gptq=args.disable_gptq,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gptq:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gptq:
            print(i, name)
            print("Quantizing ...")
            info = gptq[name].fasterquant(
                percdamp=args.percdamp,
                blocksize=args.blocksize,
                layeri=i,
                blockname=name,
            )
            gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    def list_of_ints(arg):
        return list(map(int, arg.split(",")))

    def list_of_floats(arg):
        return list(map(float, arg.split(",")))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `llama-7b`."
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="path to the model to load; for example `/home/models/opt-6.7b`.",
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "quant_method",
        type=str,
        choices=["col-hbraq", "row-hbraq"],
        help="quantization method; 'hbraq' is the method used in HBLLM",
    )
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--salient_metric",
        type=str,
        choices=["l1", "l2"],
        default="l2",
        help="Metric used to evaluate column-wise weight saliency for grouping. Choose between 'l1' or 'l2' norm. Default is 'l2'."
    )
    parser.add_argument(
        "--group_partition",
        type=str,
        choices=["global", "row"],
        help="granularity of group quantization; include global grouping and row-wise grouping strategies",
    )
    parser.add_argument(
        "--share_mean",
        action="store_true",
        help="whether to share mean when binarize",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="set the device to use for quantization.",
    )
    parser.add_argument(
        "--disable_gptq",
        action="store_true",
        help="disable GPTQ for quantization.",
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Quant all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Quant all layers with id < this."
    )
    parser.add_argument(
        "--quant_only",
        type=str,
        default="",
        help="Quant only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the model; for example `/home/models/opt-6.7b`.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )

    args = parser.parse_args()
    groupsize = args.blocksize

    device = args.device
    if args.load_quantized:
        model = get_model(args.model, args.model_path)
        model.eval()
    else:  # braq
        model = get_model(args.model, args.model_path)
        model.eval()
        tick = time.time()
        dataloader, testloader = get_loaders(
            name=args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            model_path=args.model_path,
            seqlen=model.seqlen,
        )
        quant_sequential(model, dataloader, device)
        print("quantization time:", time.time() - tick, "s")

    for dataset in ["c4", "wikitext2", "ptb"]:
        dataloader, testloader = get_loaders(
            dataset,
            seed=args.seed,
            seqlen=model.seqlen,
            model=args.model,
            model_path=args.model_path,
        )
        print(dataset)
        if "opt" in args.model:
            from eval_ppl_utils import opt_eval

            opt_eval(model, testloader, device, dataset, args.log_wandb)
        elif "llama" in args.model:
            from eval_ppl_utils import llama_eval

            llama_eval(model, testloader, device, dataset, args.log_wandb)

    if args.save:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        model.save_pretrained(args.save_path)
        tokenizer = get_tokenizer(args.model_path)
        tokenizer.save_pretrained(args.save_path)
