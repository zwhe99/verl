# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import os
import warnings

import torch
from megatron.core import dist_checkpointing
from megatron.core import parallel_state as mpu
from megatron.core.dist_checkpointing.serialization import StrictHandling
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoConfig, AutoModelForCausalLM

from verl.models.mcore import hf_to_mcore_config
from verl.utils.megatron_utils import get_model


def _init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_path", type=str, required=True, help="The path for the huggingface model")
    parser.add_argument("--output_path", type=str, required=True, help="The path for the output mcore model")
    parser.add_argument("--use_cpu_initialization", action="store_true", help="Whether to use cpu initialization")
    parser.add_argument("--test", action="store_true", help="Whether to test the conversion")
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to trust remote code")
    args = parser.parse_args()
    return args


class MegatronConfig:
    def __init__(self):
        self.params_dtype = torch.bfloat16


class ModelConfig:
    def __init__(self):
        self.path = None


class Config:
    def __init__(self):
        self.model = ModelConfig()


def test_conversion(megatron_model_provider, tfconfig, output_path, model):
    ########### test ###########
    # load model
    model_test = get_model(
        model_provider_func=megatron_model_provider,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=True,
        transformer_config=tfconfig,
    )
    ref_state_dict = model_test[0].module.sharded_state_dict()
    dist_checkpointing.load(ref_state_dict, output_path, strict=StrictHandling.ASSUME_OK_UNEXPECTED)

    dut_state_dict = model[0].module.state_dict()
    for name in dut_state_dict.keys():
        if dut_state_dict[name] is None:
            print(f"[Warning] {name} is none in dut_state_dict")
            continue
        dut_data = dut_state_dict[name].data
        if name in ref_state_dict:
            ref_data = ref_state_dict[name].data
            assert dut_data.shape == ref_state_dict.shape, f"{name=} {dut_data.shape=} {ref_data.shape=}"
            assert (dut_data == ref_data).all(), f"{name} is not equal"
            print(f"{name} is equal")
        else:
            print(f"[Warning] {name} is not in ref_state_dict")
    for name in ref_state_dict.keys():
        if ref_state_dict[name] is None:
            print(f"[Warning] {name} is none in ref_state_dict")
            continue
        ref_data = ref_state_dict[name].data
        if name in dut_state_dict:
            dut_data = dut_state_dict[name].data
            assert dut_data.shape == ref_data.shape, f"{name=} {dut_data.shape=} {ref_data.shape=}"
            assert (dut_data == ref_data).all(), f"{name} is not equal"
            print(f"{name} is equal")
        else:
            print(f"[Warning] {name} is not in dut_state_dict")
    print("Conversion test passed!")


def convert_checkpoint_from_transformers_to_megatron(hf_model, model, hf_config):
    num_attention_heads = hf_config.num_attention_heads
    num_key_value_heads = hf_config.num_key_value_heads
    hidden_dim = hf_config.hidden_size
    head_dim = getattr(hf_config, "head_dim", hidden_dim // num_attention_heads)
    if num_attention_heads != num_key_value_heads:
        print("[WARNING] Converting GQA model")
    has_qkv_bias = getattr(hf_config, "qkv_bias", False) or getattr(hf_config, "attention_bias", False)
    has_share_expert = getattr(hf_config, "shared_expert_intermediate_size", None)
    with torch.no_grad():
        model.embedding.word_embeddings.weight.copy_(hf_model.model.embed_tokens.weight)
        for layer, hf_layer in zip(model.decoder.layers, hf_model.model.layers):
            layer.self_attention.linear_qkv.layer_norm_weight.copy_(hf_layer.input_layernorm.weight)

            q = hf_layer.self_attn.q_proj.weight.view([num_key_value_heads, head_dim * num_attention_heads // num_key_value_heads, -1])
            k = hf_layer.self_attn.k_proj.weight.view([num_key_value_heads, head_dim, -1])
            v = hf_layer.self_attn.v_proj.weight.view([num_key_value_heads, head_dim, -1])
            qkv = torch.cat([q, k, v], dim=1).view(-1, hidden_dim).contiguous()
            layer.self_attention.linear_qkv.weight.copy_(qkv)

            if has_qkv_bias:
                q_bias = hf_layer.self_attn.q_proj.bias.view([num_key_value_heads, -1])
                k_bias = hf_layer.self_attn.k_proj.bias.view([num_key_value_heads, -1])
                v_bias = hf_layer.self_attn.v_proj.bias.view([num_key_value_heads, -1])
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=1).view(-1).contiguous()
                layer.self_attention.linear_qkv.bias.copy_(qkv_bias)

            if hasattr(hf_layer.self_attn, "q_norm"):
                layer.self_attention.q_layernorm.weight.copy_(hf_layer.self_attn.q_norm.weight.data)
                layer.self_attention.k_layernorm.weight.copy_(hf_layer.self_attn.k_norm.weight.data)

            layer.self_attention.linear_proj.weight.copy_(hf_layer.self_attn.o_proj.weight)
            layer.pre_mlp_layernorm.weight.copy_(hf_layer.post_attention_layernorm.weight)

            layer.mlp.router.weight.copy_(hf_layer.mlp.gate.weight)

            for idx, hf_expert in enumerate(hf_layer.mlp.experts):
                fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                layer.mlp.experts.linear_fc1._parameters[f"weight{idx}"].copy_(fc1_weight)
                layer.mlp.experts.linear_fc2._parameters[f"weight{idx}"].copy_(hf_expert.down_proj.weight)

            if has_share_expert:
                layer.mlp.shared_experts.gate_weight.copy_(hf_layer.mlp.shared_expert_gate.weight)
                shared_fc1_weight = torch.cat([hf_layer.mlp.shared_expert.gate_proj.weight, hf_layer.mlp.shared_expert.up_proj.weight])
                layer.mlp.shared_experts.linear_fc1.weight.copy_(shared_fc1_weight)
                layer.mlp.shared_experts.linear_fc2.weight.copy_(hf_layer.mlp.shared_expert.down_proj.weight)

        model.decoder.final_layernorm.weight.copy_(hf_model.model.norm.weight)
        model.output_layer.weight.copy_(hf_model.lm_head.weight)


@torch.no_grad()
def convert_checkpoint_from_transformers_to_megatron_dpskv3(hf_model, model, hf_config, tfconfig):
    warnings.warn("MTP model is not supported yet", stacklevel=2)

    def safe_copy(
        src_tensor: torch.Tensor,
        dst_tensor: torch.Tensor,
        skip_dtype_assert: bool = False,
    ):
        if not skip_dtype_assert:
            if src_tensor.dtype != dst_tensor.dtype:
                raise ValueError(f"Get source dtype {src_tensor.dtype}, but target dtype {dst_tensor.dtype}")
        assert src_tensor.shape == dst_tensor.shape
        dst_tensor.data.copy_(src_tensor.data)
        return src_tensor.numel()

    model.embedding.word_embeddings.weight.copy_(hf_model.model.embed_tokens.weight)
    for layer_idx, (layer, hf_layer) in enumerate(zip(model.decoder.layers, hf_model.model.layers)):
        print(layer_idx)
        layer.input_layernorm.weight.copy_(hf_layer.input_layernorm.weight)

        if hf_config.q_lora_rank is None:
            layer.self_attention.linear_q_proj.weight.copy_(hf_layer.self_attn.q_proj.weight)
        else:
            layer.self_attention.linear_q_down_proj.weight.copy_(hf_layer.self_attn.q_a_proj.weight)
            layer.self_attention.linear_q_up_proj.weight.copy_(hf_layer.self_attn.q_b_proj.weight)
            layer.self_attention.linear_q_up_proj.layer_norm_weight.copy_(hf_layer.self_attn.q_a_layernorm.weight)

        layer.self_attention.linear_kv_down_proj.weight.copy_(hf_layer.self_attn.kv_a_proj_with_mqa.weight)
        layer.self_attention.linear_kv_up_proj.weight.copy_(hf_layer.self_attn.kv_b_proj.weight)
        layer.self_attention.linear_kv_up_proj.layer_norm_weight.copy_(hf_layer.self_attn.kv_a_layernorm.weight)
        layer.self_attention.linear_proj.weight.copy_(hf_layer.self_attn.o_proj.weight)

        if not hasattr(layer.mlp, "router"):
            layer.mlp.linear_fc1.layer_norm_weight.copy_(hf_layer.post_attention_layernorm.weight)
            layer.mlp.linear_fc1.weight.copy_(torch.cat([hf_layer.mlp.gate_proj.weight, hf_layer.mlp.up_proj.weight]))
            layer.mlp.linear_fc2.weight.copy_(hf_layer.mlp.down_proj.weight)
        else:
            layer.mlp.router.weight.copy_(hf_layer.mlp.gate.weight)
            # NOTE: the e_score_correction_bias in mcore model will be initialized with bfloat16 and \
            # recover to fp32 in the first forward. There is always a diff in the bias between two models (~0.3%)
            safe_copy(hf_layer.mlp.gate.e_score_correction_bias, layer.mlp.router.expert_bias, skip_dtype_assert=True)
            if tfconfig.moe_grouped_gemm:
                for i, hf_expert in enumerate(hf_layer.mlp.experts):
                    fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                    linear_fc1_weighti = getattr(layer.mlp.experts.linear_fc1, "weight" + str(i))
                    linear_fc1_weighti.copy_(fc1_weight)
                    linear_fc2_weighti = getattr(layer.mlp.experts.linear_fc2, "weight" + str(i))
                    linear_fc2_weighti.copy_(hf_expert.down_proj.weight)
            else:
                for i, hf_expert in enumerate(hf_layer.mlp.experts):
                    expert = layer.mlp.experts.local_experts[i]
                    fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                    expert.linear_fc1.weight.copy_(fc1_weight)
                    expert.linear_fc2.weight.copy_(hf_expert.down_proj.weight)
            layer.pre_mlp_layernorm.weight.copy_(hf_layer.post_attention_layernorm.weight)
            shared_fc1_weight = torch.cat([hf_layer.mlp.shared_experts.gate_proj.weight, hf_layer.mlp.shared_experts.up_proj.weight])
            layer.mlp.shared_experts.linear_fc1.weight.copy_(shared_fc1_weight)
            layer.mlp.shared_experts.linear_fc2.weight.copy_(hf_layer.mlp.shared_experts.down_proj.weight)

        model.decoder.final_layernorm.weight.copy_(hf_model.model.norm.weight)
        if not hf_config.tie_word_embeddings:
            model.output_layer.weight.copy_(hf_model.lm_head.weight)


def convert_hf_to_mcore(hf_model_path, output_path, use_cpu_initialization=False, test=False, trust_remote_code=False):
    os.makedirs(output_path, exist_ok=True)
    if len(os.listdir(output_path)) > 0 and not test:
        print(f"Output path {output_path} is not empty, skipping conversion")
        return

    # init torch distributed and mpu
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(0)

    # init hf config
    hf_config = AutoConfig.from_pretrained(hf_model_path)
    print(hf_config, flush=True)

    cfg = Config()
    cfg.model.path = hf_model_path
    tfconfig = hf_to_mcore_config(hf_config, torch.bfloat16)
    tfconfig.use_cpu_initialization = use_cpu_initialization
    tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

    # init megatron model
    def megatron_model_provider(pre_process, post_process):
        from verl.models.mcore import init_mcore_model

        parallel_model = init_mcore_model(
            tfconfig,
            hf_config,
            pre_process,
            post_process,
            share_embeddings_and_output_weights=tie_word_embeddings,
            value=False,
        )
        return parallel_model

    model = get_model(
        model_provider_func=megatron_model_provider,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
        transformer_config=tfconfig,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    # init hf model
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code)
    hf_state_dict = hf_model.state_dict()

    # load hf state dict to megatron model
    if "Qwen2MoeForCausalLM" in hf_config.architectures:
        convert_checkpoint_from_transformers_to_megatron(hf_model, model[0].module, hf_config)
    elif "DeepseekV3ForCausalLM" in hf_config.architectures:
        convert_checkpoint_from_transformers_to_megatron_dpskv3(hf_model, model[0].module, hf_config, tfconfig=tfconfig)
    elif "Qwen3MoeForCausalLM" in hf_config.architectures:
        convert_checkpoint_from_transformers_to_megatron(hf_model, model[0].module, hf_config)
    else:
        assert not use_cpu_initialization, "use_cpu_initialization is only supported for MoE model"
        from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel

        load_state_dict_to_megatron_gptmodel(
            state_dict=hf_state_dict,
            wrapped_models=model,
            config=hf_config,
            params_dtype=torch.bfloat16,
            is_value_model=False,
        )

    megatron_state_dict = model[0].module.sharded_state_dict()
    del hf_state_dict, hf_model

    # save megatron model
    if len(os.listdir(output_path)) == 0:
        dist_checkpointing.save(megatron_state_dict, output_path, sharded_strategy=None, async_sharded_save=False)
    if test:
        test_conversion(megatron_model_provider, tfconfig, output_path, model)


if __name__ == "__main__":
    args = _init_args()
    convert_hf_to_mcore(args.hf_model_path, args.output_path, args.use_cpu_initialization, args.test, args.trust_remote_code)
