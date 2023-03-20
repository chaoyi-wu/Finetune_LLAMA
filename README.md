## S1：
首先安装进入Python_Package安装相关库文件，重点在于peft包transformers包的安装，建议先使用pip安装online package保证依赖包都顺利安装，再pip install -e .本地安装替换。
**注意**：pytorch包务必使用conda安装！sentencepiece包勿遗漏。

## S2：
进入LLAMA_Model下载模型参数 https://huggingface.co/decapoda-research/llama-7b-hf 或者官网下载llama，使用convert_to_ds_params.py进行处理。

## S3：
进入Data_sample按照示例处理数据。

## S4：
修改finetune_pp.py或finetune_pp_peft.py相关参数（前者为整个网络参数均进行finetune，后者参考lora进行部分参数finetune），指定GPU（直接命令行，os.environment暂时不知道为什么无效），即可进行训练。 以batch4为例，peft版大概4张A100每张会使用60G显存，一个step耗时16s前后，pp版则8张A100每张会使用70G前后，一个step耗时65s。

## S5：
参考test_sample.py进行测试，注意测试时，尽量避免使用多卡。

## LLAMA模型下载地址：
预训练模型下载 https://huggingface.co/nyanko7/LLaMA-7B

迅雷地址 [magnet:?xt=urn:btih:ZXXDAUWYLRUXXBHUYEMS6Q5CE5WA3LVA&dn=LLaMA]（未测试） 

## Acknowledge:
参考 Minimal LLaMA https://github.com/zphang/minimal-llama 实现，主要修复了部分bug。

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
