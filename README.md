## Abstract

We introduce HBLLM, a wavelet-enhanced high-fidelity $1$-bit post-training quantization method for Large Language Models (LLMs). By leveraging Haar wavelet transforms to enhance expressive capacity through frequency decomposition, HBLLM significantly improves quantization fidelity while maintaining minimal overhead. This approach features two innovative structure-aware grouping strategies: (1) frequency-aware multi-parameter intra-row grouping and (2) $\ell_2$-norm-based saliency-driven column selection. For non-salient weights, a shared mean is employed across quantization groups within each frequency band to optimize storage efficiency. Experiments conducted on the OPT and LLaMA models demonstrate that HBLLM achieves state-of-the-art performance in $1$-bit quantization, attaining a perplexity of $6.71$ perplexity on LLaMA$2$-$13$B with an average weight storage of only $1.08$ bits.

## Dependencies

* `torch`: tested on v2.2.2+cu118
* `transformers`: tested on v4.35.0 (the LLaMa integration currently requires a main install from source and `sentencepiece`)
* `datasets`: tested on v2.14.6

All binarization processes and experiments were run on a single 80GB NVIDIA A100. However, all the process can also be conducted on a single 24GB NVIDIA 3090 Ti when the model's parameter is under 70B.

## LLMs Binarization

#### Binarization for OPT families

##### Row-wise Haar transform (row-hbraq)
```
python3 run.py opt-1.3b /home/models/opt-1.3b c4 row-hbraq --blocksize 128 --salient_metric l2 --group_partition row 
```
or
##### Column-wise Haar transform (col-hbraq)
```
python3 run.py opt-1.3b /home/models/opt-1.3b c4 col-hbraq --blocksize 128 --salient_metric l2 --group_partition row 
```

#### Binarization for LLaMA families

```
python3 run.py llama2-7b /home/models/llama2-7b c4 row-hbraq --blocksize 128 --salient_metric l2 --group_partition row 
```
##### use shared_mean strategy
or
```
python3 run.py llama2-7b /home/models/llama2-7b c4 row-hbraq --blocksize 128 --salient_metric l2 --group_partition row --share_mean
```

## Results


## Related Project
[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://github.com/IST-DASLab/gptq)

[PB-LLM: Partially Binarized Large Language Models](https://github.com/hahnyuan/PB-LLM)


[BiLLM: Pushing the Limit of Post-Training Quantization for LLMs](https://github.com/Aaronhuang-778/BiLLM)

