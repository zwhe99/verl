# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import ray
from tqdm import tqdm

def is_ray_remote_function(func):
    return hasattr(func, 'remote') and callable(func.remote)

class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def __call__(self, data: DataProto):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        if is_ray_remote_function(self.compute_score):
            return self._call_reward_ray(data)
        else:
            return self._call_reward(data)

    def _call_reward_ray(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        prompt_length = data.batch['prompts'].shape[-1]

        # get data source list
        data_source_lst = [data[i].non_tensor_batch['data_source'] for i in range(len(data))]

        # get solution str list
        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        valid_response_ids = [
            response_ids[i][:valid_response_length[i]]
            for i in range(len(data))
        ]
        solution_str_lst = [
            self.tokenizer.decode(valid_response_ids[i])
            for i in range(len(data))
        ]

        # get ground truth list
        ground_truth_lst = [
            data[i].non_tensor_batch['reward_model']['ground_truth']
            for i in range(len(data))
        ]

        # get extra info list
        extra_info_lst = [
            data[i].non_tensor_batch.get('extra_info', None)
            for i in range(len(data))
        ]

        # compute reward
        reward_future_lst = [self.compute_score.remote(
            data_source=data_source_lst[i],
            solution_str=solution_str_lst[i],
            ground_truth=ground_truth_lst[i],
            extra_info=extra_info_lst[i],
        ) for i in range(len(data))]
        score_lst = ray.get(reward_future_lst)

        # fill reward tensor
        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i] - 1] = score_lst[i]

        # print to console
        already_print_data_sources = {}
        for i in range(len(data)):
            data_item = data[i]
            data_source = data_item.non_tensor_batch['data_source']
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(solution_str_lst[i])

        return reward_tensor

    def _call_reward(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            solution_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor
