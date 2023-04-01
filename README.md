## S1：
首先安装进入Python_Package安装相关库文件，重点在于peft包transformers包的安装，建议先使用pip安装online package保证依赖包都顺利安装，再```pip install -e .```本地安装替换。
**注意**：pytorch包务必使用conda安装！sentencepiece包勿遗漏。

## S2：
进入LLAMA_Model下载模型参数 https://huggingface.co/decapoda-research/llama-7b-hf 或者官网下载llama，使用convert_to_ds_params.py进行处理。

## S3：
进入Data_sample按照示例处理数据。

## S4：
修改finetune_pp.py或finetune_pp_peft.py相关参数（前者为整个网络参数均进行finetune，后者参考lora进行部分参数finetune），指定GPU（直接命令行，os.environment暂时不知道为什么无效），即可进行训练。 以batch4（max-seq-length=2048）为例，peft版大概4张A100每张会使用60G显存，一个step耗时5s前后，pp版则8张A100每张会使用70G前后，一个step耗时30s。使用trainer实现单机多卡并行后，可以大幅提速，4张A100每张使用70G显存，可以以batch16（max-seq-length=512）进行训练，一个step耗时7s前后。

## S5：
参考test_sample.py进行测试，注意测试时，尽量避免使用多卡。

## 更新3.30：
增加两个_trainer.py, 利用transformers.trainer简单实现单机多卡并行，使用fsdp解决了单卡爆卡的问题，训练速度显著加快（To do：lora版本还没有有效使用fsdp，目前可以运行单纯是因为lora优化参数少单卡不会爆。。）

## 更新3.31：
修复了lora与fsdp不能并存的问题，目前可以支持13B的训练，并且在lora版本下11s可以处理2048个tokens，在完全finetune的情况下也可以做到60s处理2048个tokens。并且可以支持33B（lora）版的训练，但是速度还是比较慢，大概200s单卡可以处理完2048个tokens。

## 更新4.1：
Not Fooling！使用deepspeed替换了fsdp，现在可以finetune 33B（lora）可以达到16s完成单卡1536（3batch of 512）一次step。

关于deepspeed的库安装，首先pytorch环境还是必须得conda安装，其次使用conda安装gcc6 ```conda install -c omgarcia gcc-6```， 然后更新一下gcc的动态库 ```conda install -c anaconda libstdcxx-ng``` 最后```git clone https://github.com/microsoft/DeepSpeed```， 使用```DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e .```安装。

## LLAMA模型下载地址：

迅雷地址 [magnet:?xt=urn:btih:ZXXDAUWYLRUXXBHUYEMS6Q5CE5WA3LVA&dn=LLaMA]（未测试） 

## Acknowledge:
参考 Minimal LLaMA https://github.com/zphang/minimal-llama 实现，主要修复了部分bug。

参考alpaca https://github.com/tatsu-lab/stanford_alpaca 加入fsdp。

参考LMFLow https://github.com/OptimalScale/LMFlow/tree/main/src/lmflow 加入deepspeed模块。

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
