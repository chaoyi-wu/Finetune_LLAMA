import argparse
import json
import numpy as np
import random
import tqdm.auto as tqdm

import datasets
import transformers


def read_jsonl(path):
    # Manually open because .splitlines is different from iterating over lines
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str,default='/nvme/zhangruipeng/wuchaoyi/wuchaoyi/llama/tokenizer')
    parser.add_argument("--jsonl_path", type=str,default='/nvme/zhangruipeng/wuchaoyi/wuchaoyi/BioMedLM/finetune/mc/data/medqa_usmle_hf/train.json')
    parser.add_argument("--save_path", type=str,default='/nvme/zhangruipeng/wuchaoyi/minimal-llama/UMLSE_TOKENIZED')
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()
    tokenizer = transformers.LlamaTokenizer.from_pretrained(args.tokenizer_path)
    all_tokenized = []
    ending_names = ["ending0", "ending1", "ending2", "ending3"]
    for elem in tqdm.tqdm(read_jsonl(args.jsonl_path)):
        instuction = "You are a doctor. According to the question, make a choice. Q: "
        reflect = {0:'A',1:'B',2:'C',3:'D'}
        sentence = instuction + elem["sent1"]
        for j,end in enumerate(ending_names):
            sentence = sentence + ' ' + reflect[j] + ': ' + elem[end]
        sentence = sentence + " The Answer is: " + reflect[elem['label']]
        all_tokenized.append(tokenizer.encode(sentence))
    random.shuffle(all_tokenized)

    all_tokens = [1] + [
        tok
        for row in all_tokenized
        for tok in row + [tokenizer.eos_token_id, tokenizer.bos_token_id]
    ]

    truncated_tokens = all_tokens[:(len(all_tokens) // args.max_seq_length) * args.max_seq_length]
    arr = np.array(truncated_tokens).reshape(-1, args.max_seq_length)
    ds = datasets.Dataset.from_dict({"input_ids": arr})
    ds.save_to_disk(args.save_path)
    print(f"Generated {arr.shape[0]} samples.")


if __name__ == "__main__":
    main()
