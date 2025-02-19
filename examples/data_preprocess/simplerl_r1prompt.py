"""
Preprocess the simplerl dataset to parquet format
"""

import os
import datasets
import argparse

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

data_source = 'zwhe99/simplerl'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/simplerl_r1prompt')
    args = parser.parse_args()

    train_dataset = datasets.load_dataset(data_source, split='train')

    def process_fn_train(example, idx):
        data = {
            "data_source": "simplerl_r1prompt",
            "prompt": example['messages'],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": example['answer']
            },
            "extra_info": {
                'split': 'train',
                'index': idx,
                'answer': example['answer'],
                "question": example['problem'],
            }
        }
        return data

    def process_fn_test(example, idx):
        data = {
            "data_source": "simplerl_r1prompt",
            "prompt": example['messages'],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": example['answer']
            },
            "extra_info": {
                'split': 'test',
                'index': idx,
                'answer': example['answer'],
                "question": example['problem'],
            }
        }
        return data

    train_dataset = train_dataset.map(function=process_fn_train, with_indices=True)
    test_dataset = train_dataset.map(function=process_fn_test, with_indices=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))