import os
import datasets
import json
import argparse
from datasets import Dataset

def read_jsonl(file):
    """
    Read a JSONL file.

    Args:
        file (str): The path to the JSONL file.

    Returns:
        List[dict]: A list of dictionaries, each representing a sample.
    """
    if not os.path.exists(file):
        return []

    with open(file, "r", encoding="utf-8") as f:
        # Read all lines at once instead of line by line
        lines = f.readlines()

        # Use list comprehension with json.loads
        return [json.loads(line) for line in lines]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--local_dir', default='~/data/orz57k')
    args = parser.parse_args()

    ori_data = read_jsonl(args.data_path)

    # process the dataset
    data = []
    for idx, od in enumerate(ori_data):
        data.append({
            "data_source": "v2hq-v",
            "prompt": [
                {
                    "role": "system",
                    "content": r"Please reason step by step, and put your final answer within \boxed{}."
                },
                {
                    "role": "user",
                    "content": od["problem"]
                },                
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": od["expected_answer"]
            },
            "extra_info": {
                'split': 'train',
                'index': idx,
                'answer': od["expected_answer"],
                "question": od["problem"],
            }
        })

    # add simplerl data
    simplerl_dataset = datasets.load_dataset('zwhe99/simplerl', split='train')
    for idx, sd in enumerate(simplerl_dataset):
        data.append({
            "data_source": "v2hq-v",
            "prompt": sd['messages'],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": sd['answer']
            },
            "extra_info": {
                'split': 'train',
                'index': idx,
                'answer': sd['answer'],
                "question": sd['problem'],
            }
        })

    train_dataset = Dataset.from_list(data)
    test_dataset = Dataset.from_list(data)
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
