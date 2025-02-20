"""
Preprocess the orz57k dataset to parquet format
"""

import os
import requests
import argparse
from datasets import Dataset

data_url = "https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_57k_collected.json"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/orz57k')
    args = parser.parse_args()

    # download the dataset
    try:
        response = requests.get(data_url)
        response.raise_for_status()
        ori_data = response.json()
    except Exception as e:
        print(f"Error downloading the dataset: {e}")
        exit(1)

    # process the dataset
    data = []
    for idx, od in enumerate(ori_data):
        assert len(od) == 2
        assert od[0]["from"] == "human" and od[1]["from"] == "assistant"
        data.append({
            "data_source": "orz57k",
            "prompt": [
                {
                    "role": "system",
                    "content": r"Please reason step by step, and put your final answer within \boxed{}."
                },
                {
                    "role": "user",
                    "content": od[0]["value"]
                },
                
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": od[1]["ground_truth"]["value"]
            },
            "extra_info": {
                'split': 'train',
                'index': idx,
                'answer': od[1]["ground_truth"]["value"],
                "question": od[0]["value"],
            }
        })

    train_dataset = Dataset.from_list(data)
    test_dataset = Dataset.from_list(data)
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
