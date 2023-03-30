import argparse
import os
import json
import math
import tqdm.auto as tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import datasets
import transformers
from finetune_peft import get_peft_config, CastOutputToFloat, save_tunable_parameters
from transformers import Trainer
from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)
import os
    
def write_json(x, path):
    with open(path, "w") as f:
        f.write(json.dumps(x))
    
def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        
IGNORE_INDEX = -100
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="./LLAMA_Model/llama-13b-hf")
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)

@dataclass
class DataArguments:
    Train_dataset_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

class DatasetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return dict(input_ids=self.dataset[idx]["input_ids"], labels=self.dataset[idx]["input_ids"])
    
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("Setup Data")
    Train_dataset = DatasetDataset(datasets.load_from_disk(data_args.Train_dataset_path))
    #eval_dataset = DatasetDataset(datasets.load_from_disk(data_args.Test_dataset_path))
    
    
    print("Setup Model")
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_path,
        cache_dir=training_args.cache_dir,
    )

    print("Setup PEFT")
    peft_config = get_peft_config(peft_args=model_args)
    model = get_peft_model(model, peft_config)
    training_args.fsdp_config['fsdp_min_num_params']= 10*4096 ### 重要！lora_rank == 8 的情况下，lora矩阵刚好是8*4096的size。该参数指定了大于该数值的模块被wrap，10*4096的设置刚好保证了lora矩阵不会被wrap。
    # print("Setup optimizer")
    # opt = torch.optim.AdamW([
    #     p
    #     for p in model.parameters()
    #     if p.requires_grad
    # ], lr=training_args.learning_rate)
    
    #lrsche = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 4, eta_min=0, last_epoch=-1, verbose=False) 
    
    trainer = Trainer(model=model, 
                      train_dataset = Train_dataset, 
                      #eval_dataset = eval_dataset,
                      #optimizers = [opt,lrsche],
                      args=training_args,
                      )
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
      


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetune_pp_peft_trainer.py