# 微调LLAMA的中文指南
本项目旨在引导中文用户微调Large Language Model（[LLAMA](https://arxiv.org/abs/2302.13971)），整合了目前多个框架（[Minimal LLaMA](https://github.com/zphang/minimal-llama)、[Alpaca](https://github.com/tatsu-lab/stanford_alpaca)、[LMFlow](https://github.com/OptimalScale/LMFlow/tree/main/src/lmflow)），尽量避免不必要的封装，保证代码可读性。
## S1：
进入Python_Package安装相关peft包和transformers包。

建议先使用pip安装online package保证依赖包都顺利安装，再```pip install -e .```本地安装替换。

**注意**：\
pytorch包务必使用conda安装！```conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia```\
不要忘记安装sentencepiece！```pip install sentencepiece```\

## S2：
进入LLAMA_Model下载模型参数 https://huggingface.co/decapoda-research/llama-7b-hf 或者官网下载llama，使用convert_to_ds_params.py进行处理。

## S3：
进入Data_sample按照示例处理数据。

## S4：
修改finetune_pp.py或finetune_pp_peft.py相关参数（前者为整个网络参数均进行finetune，后者参考lora进行部分参数finetune），指定GPU，即可进行训练。

**注意**：finetune_pp.py与finetune_pp_peft.py无多卡加速，训练速度缓慢，但可以有效避免oom，适合debug。加速请参考**FSDP多卡并行**与**DeepSpeed多卡并行**。

## S5：
参考test_sample.py进行测试，测试时，尽量避免使用多卡。

## FSDP多卡并行：
finetune_pp_peft_trainer_lora.py与finetune_pp_peft_trainer.py种利用transformers.trainer简单实现单机多卡并行，使用fsdp解决了单卡爆卡的问题，训练速度显著加快。

## DeepSpeed多卡并行：
DeepSpeed的库安装：\
```conda install -c omgarcia gcc-6```，使用conda安装gcc6\
```conda install -c anaconda libstdcxx-ng``` ，更新gcc的动态库\
```git clone https://github.com/microsoft/DeepSpeed```，下载DS库\
```DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e .```完成安装。\
cuda环境有问题可以参考这个[issue#2684](https://github.com/microsoft/DeepSpeed/issues/2684)

```sh finetune_pp_peft_trainer_deepspeed.sh```进行训练，传入参数```--lora_used True(False)```控制是否使用lora。

DS版本支持33B（lora）llama快速进行finetune，训练时长与[LMFlow](https://github.com/OptimalScale/LMFlow/tree/main/src/lmflow)一致。

## 训练时长统计：
在4.8M PMCOA papers上统计各种训练设置的耗时。

训练时默认采用8张A100，每次对paper随机抽取一段512 tokens长度的句子进行训练，等价于一个epoch会处理**2.5B**tokens。
| Statistic on S2ORC (4.8M PMCOA papers) |            |               |            |
| -------------------------------------- | ---------- | ------------- | ---------- |
| Model_Size                             | Batch_Size | Accelerate Strategy      | Time/epoch |
| 13B                                    | 384        | DS*（Opt&Par） | ~122h      |
| 7B                                     | 768        | DS（Opt&Par） | ~100h      |
| 7B                                     | 128        | DS（Opt&Par） | ~100h      |
| 7B                                     | 384        | DS（Opt） | ~90h       |
| 7B                                     | 384        | FSDP_no_cpu   | ~35h       |
| 7B                                     | 128        | FSDP_no_cpu   | ~36h       |
> DS(Opt&Par):optimizer and persistent parameters offloaded to cpu\
> DS(Opt):optimizer offloaded to cpu\
> FSDP_no_cpu: No cpu involved\
> 注：cpu参与会导致训练速度变慢，但规模上去后，比如13B，必须CPU参与才可以完成多卡并行。表中上标*代表必须采用这种加速策略才能避免OOM。

## PMC论文训练结果：
我们在S2ORC数据集提供的4.8M PMC paper上finetune了LLaMA-7B，得到了PMC_LLaMA_7B([Hugging Face URL](https://huggingface.co/chaoyi-wu/PMC_LLAMA_7B))

整体的训练设置如下：
* Epochs: 5
* Batch size: 128
* Cutoff length: 512
* Learning rate: 2e-5

每个epoch我们都从每篇paper中抽取512 tokens用来训练。

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
