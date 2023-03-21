import transformers
import os
import json
import tqdm
import argparse
import torch 
import math 
from peft import get_peft_model
from finetune_peft import get_peft_config, CastOutputToFloat

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)
def read_jsonl(path):
    # Manually open because .splitlines is different from iterating over lines
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)
            
def Setup_model(model_path, peft_flag = True, peft_arg = None):
    # The auto/balance balancing strategy doesn't seem to work correctly,
    # so we manually compute the mappings.
    print("Setup Model")
    num_layers = read_json(os.path.join(model_path, "config.json"))["num_hidden_layers"]
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
        model_path,
        #load_in_8bit=True,
        device_map=device_map,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    if peft_flag:
        print("Setup PEFT")
        model.lm_head = CastOutputToFloat(model.lm_head)
        
        peft_config = get_peft_config(peft_args=peft_arg)
        model = get_peft_model(model, peft_config)

        for layer_i in range(len(allocations)):
            device = model.base_model.model.model.layers[layer_i].self_attn.q_proj.weight.device
            model.base_model.model.model.layers[layer_i].self_attn.q_proj.lora_B.to(device)
            model.base_model.model.model.layers[layer_i].self_attn.q_proj.lora_A.to(device)
            model.base_model.model.model.layers[layer_i].self_attn.v_proj.lora_B.to(device)
            model.base_model.model.model.layers[layer_i].self_attn.v_proj.lora_A.to(device)
        
        if os.path.exists(peft_arg.peft_path + '/latest.json'):
            start = read_json(peft_arg.peft_path + '/latest.json')["latest_step"]
            model.load_state_dict(
                torch.load(os.path.join(os.path.join(args.peft_path, f"params-{start + 1:06d}.p"))), strict=False)
    return model
        
if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/nvme/zhangruipeng/wuchaoyi/Finetune_llama_by_wucc/LLAMA_Model/llama-7b")
    parser.add_argument("--tokenizer_path", type=str, default="/nvme/zhangruipeng/wuchaoyi/wuchaoyi/llama/tokenizer")
    parser.add_argument("--test_path", type=str, default="/nvme/zhangruipeng/zhangxiaoman/code/2023_MIMIC/A1_DATA/data_file/valid.json")
    parser.add_argument("--peft_path", type=str, default="/nvme/zhangruipeng/wuchaoyi/Finetune_llama_by_wucc/Fine_Tuning_Results/test")
    '''
    Peft_path is used for loading peft parameters. If set peft_flag True, clarify it and set model_path as the original weight of llama provide by meta.
    '''
       
    parser.add_argument("--peft_mode", type=str, default="lora")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--num_virtual_tokens", type=int, default=32)
    parser.add_argument("--mapping_hidden_dim", type=int, default=1024)

    args = parser.parse_args()

    tokenizer = transformers.LlamaTokenizer.from_pretrained(args.tokenizer_path)    
    model = Setup_model(model_path = args.model_path, peft_flag = True, peft_arg = args)
    
    model.eval()
    for elem in tqdm.tqdm(read_jsonl(args.test_path)):
        i=0
        instuction = "Predict what will happen next in the hospital based on the status of this patient: "
        sentence = instuction + elem["sent"]
        sentence = sentence + "The next step is: " + elem["label"]
        batch = tokenizer(
            sentence,
            return_tensors="pt", 
            add_special_tokens=False
        )
        batch = {k: v.cuda() for k, v in batch.items()}
        
        with torch.no_grad():
            generated = model.generate(inputs = batch["input_ids"], max_length=300)
            print('...............................')
            print('input sentence:',sentence)
            print('model predict: ',tokenizer.decode(generated[0]))
            i+=1
        if i == 5:
            break