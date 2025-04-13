import os
import datasets
import json
import argparse
import datasets
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
        for r1_id in range(0, 3):
            r1_response = od["model_solutions"]["deepseek-ai/DeepSeek-R1"][0].strip()
            if r1_response.startswith("<think>"):
                r1_response = r1_response[len("<think>"):]

            if "</think>" in r1_response and r1_response.count("</think>") == 1:
                r1_response_reasoning_content = r1_response.split("</think>")[0].strip()
                r1_response_content = r1_response.split("</think>")[1].strip()
            else:
                r1_response_reasoning_content = r1_response
                r1_response_content = ""

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
                },
                "r1_content": r1_response_content,
                "r1_reasoning_content": r1_response_reasoning_content,
                "r1": f"{r1_response_reasoning_content} </think> {r1_response_content}",
                "r1_with_ans_label": f"{r1_response_reasoning_content} </think> <answer> {r1_response_content} </answer>",
            })

    # add simplerl data
    simplerl_dataset = datasets.load_dataset('zwhe99/simplerl', split='train')
    for idx, sd in enumerate(simplerl_dataset):
        r1_response = sd["r1_response"].strip()
        if r1_response.startswith("<think>"):
            r1_response = r1_response[len("<think>"):]

        if "</think>" in r1_response and r1_response.count("</think>") == 1:
            r1_response_reasoning_content = r1_response.split("</think>")[0].strip()
            r1_response_content = r1_response.split("</think>")[1].strip()
        else:
            r1_response_reasoning_content = r1_response
            r1_response_content = ""

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
            },
            "r1_content": r1_response_content,
            "r1_reasoning_content": r1_response_reasoning_content,
            "r1": f"{r1_response_reasoning_content} </think> {r1_response_content}",
            "r1_with_ans_label": f"{r1_response_reasoning_content} </think> <answer> {r1_response_content} </answer>",
        })

    # test data
    test_dataset = datasets.load_dataset('zwhe99/MATH', split='math500')
    def process_fn_test(example, idx):
        data = {
            "data_source": "math500",
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
                "split": "math500",
                "index": idx,
                "answer": example["expected_answer"],
                "question": example["problem"],
            },
        }
        return data


    train_dataset = Dataset.from_list(data)
    test_dataset = test_dataset.map(function=process_fn_test, with_indices=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
