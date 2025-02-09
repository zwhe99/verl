"""
Preprocess the simplerl dataset to parquet format
"""

import os
import datasets
import argparse

data_source = 'zwhe99/simplerl'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/simplerl')
    args = parser.parse_args()

    train_dataset = datasets.load_dataset(data_source, split='train')

    def process_fn(example, idx):
        data = {
            "data_source": data_source,
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

    train_dataset = train_dataset.map(function=process_fn, with_indices=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
