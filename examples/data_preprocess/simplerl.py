"""
Preprocess the simplerl dataset to parquet format
"""

import os
import datasets
import argparse

train_data_source = "zwhe99/simplerl"
train_split = "train"
test_data_source = "zwhe99/MATH"
test_split = "math500"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/simplerl')
    args = parser.parse_args()

    train_dataset = datasets.load_dataset(train_data_source, split=train_split)
    test_dataset = datasets.load_dataset(test_data_source, split=test_split)

    def process_fn_train(example, idx):
        data = {
            "data_source": train_data_source,
            "prompt": [
                {
                    "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                    "role": "system",
                },
                {
                    "content": example["problem"],
                    "role": "user",
                },
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": example["answer"]},
            "extra_info": {
                "split": train_split,
                "index": idx,
                "answer": example["answer"],
                "question": example["problem"],
            },
        }
        return data

    def process_fn_test(example, idx):
        data = {
            "data_source": test_split,
            "prompt": [
                {
                    "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                    "role": "system",
                },
                {
                    "content": example["problem"],
                    "role": "user",
                },
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": example["expected_answer"],
            },
            "extra_info": {
                "split": test_split,
                "index": idx,
                "answer": example["expected_answer"],
                "question": example["problem"],
            },
        }
        return data

    if not os.path.exists(os.path.join(args.local_dir)):
        os.makedirs(os.path.join(args.local_dir))

    train_dataset = train_dataset.map(function=process_fn_train, with_indices=True)
    test_dataset = test_dataset.map(function=process_fn_test, with_indices=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
