import argparse
import os
import math
import tqdm.auto as tqdm

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import datasets
import transformers
import os

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def move_to_device(*x_list, device):
    if len(x_list) > 1:
        return tuple([x.to(device) for x in x_list])
    else:
        return x_list[0].to(device)


def get_devices():
    return [
        torch.device(f"cuda:{i}")
        for i in range(torch.cuda.device_count())
    ]


def model_forward(model, inputs):
    '''
    Note that this codes are used for debugging or simple classification.
    '''
    h = inputs
    h = h.to(model.model.embed_tokens.weight.device)
    h = model.model.embed_tokens(h)
    for layer in model.model.layers:
        h = h.to(layer.input_layernorm.weight.device)
        h = layer(h)[0]
    h = h.to(model.model.norm.weight.device)
    h = model.model.norm(h)
    h = model.lm_head(h)
    return h


class DatasetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.dataset[idx]["input_ids"]),
            torch.LongTensor(self.dataset[idx]["input_ids"]),
        )


# From DeepSpeed
class RepeatingLoader:
    def __init__(self, loader):
        """Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.

        Args:
            loader (iterator): The data loader to repeat.
        """
        self.loader = loader
        self.data_iter = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,default='./LLAMA_Model/llama-7b')
    parser.add_argument("--dataset_path", type=str,default='./Data_sample/UMLSE_Train_Tokenized')
    parser.add_argument("--save_dir", type=str,default= './Fine_Tuning_Results/UMLSE_whole')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_steps", type=int, default = 1500)
    parser.add_argument("--save_interval", type=int, default = 500)
    args = parser.parse_args()

    print("Setup Data")
    dataset = datasets.load_from_disk(args.dataset_path)
    dataloader = RepeatingLoader(torch.utils.data.DataLoader(
        DatasetDataset(dataset),
        batch_size=args.batch_size,
        shuffle=True
    ))

    print("Setup Model")
    num_layers = read_json(os.path.join(args.model_path, "config.json"))["num_hidden_layers"]
    device_ids = list(range(torch.cuda.device_count()))
    device_map = {
        "model.embed_tokens": device_ids[0],
        "model.norm.weight": device_ids[-1],
        "lm_head": device_ids[-1],
    }
    allocations = [
        device_ids[i] for i in
        sorted(list(range(len(device_ids))) * math.ceil(num_layers / len(device_ids)))
    ]
    for layer_i, device_id in enumerate(allocations):
        device_map[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.mlp.down_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.mlp.up_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.input_layernorm.weight"] = device_id
        device_map[f"model.layers.{layer_i}.post_attention_layernorm.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = device_id

    model = transformers.LlamaForCausalLM.from_pretrained(
        args.model_path,
        #load_in_8bit=True,
        device_map=device_map,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    print("Setup optimizer")
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Train
    print("Start training")
    generator = iter(dataloader)
    for step in tqdm.trange(args.num_train_steps):
        input_ids, labels = next(generator)
        # logits = model_forward(model, input_ids)
        # loss = F.cross_entropy(
        #     logits.view(-1, model.config.vocab_size),
        #     labels.view(-1).to(logits.device),
        # )
        output = model(input_ids = input_ids, labels = labels)
        loss = output['loss']
        loss.backward()
        opt.step()
        if step % 1 == 0:
            print(f"Loss={loss.item():.3f}")
        actual_step = step + 1
        if actual_step % args.gradient_accumulation_steps == 0:
            opt.zero_grad()

        if actual_step % args.save_interval and actual_step != args.num_train_steps:
            model.save_pretrained(
                os.path.join(args.save_dir), f"checkpoint-{actual_step}",
                max_shard_size="500MB",
            )

    model.save_pretrained(
        os.path.join(args.save_dir), f"checkpoint-final",
        max_shard_size="500MB",
    )


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES='4,5,6,7' python finetune_pp.py