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

from collections import defaultdict

import torch
import ray
from copy import deepcopy

from verl import DataProto
from verl.utils.reward_score import default_compute_score

def is_ray_remote_function(func):
    return hasattr(func, 'remote') and callable(func.remote)

class DAPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        if is_ray_remote_function(self.compute_score):
            return self._call_reward_ray(data, return_dict)
        else:
            return self._call_reward(data, return_dict)

    def _call_reward(self, data, return_dict):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score

            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


    def _call_reward_ray(self, data, return_dict):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_length = data.batch['prompts'].shape[-1]

        # get data source list
        data_source_lst = [data[i].non_tensor_batch[self.reward_fn_key] for i in range(len(data))]

        # get prompt str list
        prompt_ids = data.batch['prompts']
        valid_prompt_length = data.batch['attention_mask'][:, :prompt_length].sum(dim=-1)
        valid_prompt_ids = [
            prompt_ids[i][-valid_prompt_length[i]:]
            for i in range(len(data))
        ]
        prompt_str_lst = [
            self.tokenizer.decode(valid_prompt_ids[i], skip_special_tokens=True)
            for i in range(len(data))
        ]

        # get solution str list
        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        valid_response_ids = [
            response_ids[i][:valid_response_length[i]]
            for i in range(len(data))
        ]
        solution_str_lst = [
            self.tokenizer.decode(valid_response_ids[i], skip_special_tokens=True)
            for i in range(len(data))
        ]

        eos_token = self.tokenizer.eos_token
        solution_str_lst = [
            s[:-len(eos_token)] if s.endswith(eos_token) else s
            for s in solution_str_lst
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
        result_lst = ray.get(reward_future_lst)

        if isinstance(result_lst[0], dict):
            score_lst = [result['score'] for result in result_lst]
            for r in result_lst:
                for k, v in r.items():
                    reward_extra_info[k].append(v)
        else:
            score_lst = result_lst

        # `score`: score from reward function
        # `reward`: consider overlong buffer
        reward_lst = deepcopy(score_lst)

        if self.overlong_buffer_cfg.enable:
            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len_lst = [valid_response_length[i] - expected_len for i in range(len(data))]
            overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            overlong_reward_lst = [min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0) for exceed_len in exceed_len_lst]
            reward_lst = [r + o for r, o in zip(reward_lst, overlong_reward_lst)]
            if self.overlong_buffer_cfg.log:
                reward_extra_info["overlong_reward"].extend(overlong_reward_lst)
                reward_extra_info["overlong"].extend([o < 0 for o in overlong_reward_lst])

        # fill reward tensor
        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i] - 1] = reward_lst[i]

        # print to console
        already_print_data_sources = {}
        for i in range(len(data)):
            data_item = data[i]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str_lst[i])
                print("[response]", solution_str_lst[i])
                print("[ground_truth]", ground_truth_lst[i])
                if isinstance(result_lst[i], dict):
                    for k, v in result_lst[i].items():
                        print(f"[{k}]", v)
                else:
                    print(f"[score]", score_lst[i])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
