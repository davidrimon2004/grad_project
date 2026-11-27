#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import json
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from videollava.model import *
from videollava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN


def _is_llava_checkpoint(model_path, model_name):
    if 'llava' in model_name.lower():
        return True
    if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
        return True
    if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        return True
    cfg_path = os.path.join(model_path, 'config.json')
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            if cfg.get('model_type') == 'llava':
                return True
        except Exception:
            pass
    return False


def _sanitize_tower_paths(cfg, load_video_tower=True):
    img_tower = getattr(cfg, 'mm_image_tower', None)
    if img_tower and img_tower.endswith('LanguageBind_Image') and not os.path.exists(img_tower) and img_tower != 'LanguageBind/LanguageBind_Image':
        warnings.warn(f"Local mm_image_tower '{img_tower}' not found; using 'LanguageBind/LanguageBind_Image'.")
        cfg.mm_image_tower = 'LanguageBind/LanguageBind_Image'
    if not load_video_tower:
        cfg.mm_video_tower = None
    else:
        vid_tower = getattr(cfg, 'mm_video_tower', None)
        if vid_tower and vid_tower.endswith('LanguageBind_Video_merge') and not os.path.exists(vid_tower) and vid_tower != 'LanguageBind/LanguageBind_Video_merge':
            warnings.warn(f"Local mm_video_tower '{vid_tower}' not found; using 'LanguageBind/LanguageBind_Video_merge'.")
            cfg.mm_video_tower = 'LanguageBind/LanguageBind_Video_merge'


def _align_fusion_modules(model, compute_dtype, zero_gate=False):
    base_model = model.get_model() if hasattr(model, 'get_model') else model
    target_device = model.device
    for attr in (
        'mm_projector',
        'early_fusion_attn',
        'early_fusion_ln',
        'early_fusion_kv_ln',
        'early_fusion_gate',
        'early_fusion_gate_linear',
    ):
        mod = getattr(base_model, attr, None)
        if mod is not None:
            mod.to(device=target_device, dtype=compute_dtype)
    if zero_gate:
        gate_mod = getattr(base_model, 'early_fusion_gate', None)
        if gate_mod is not None and hasattr(gate_mod, 'g') and isinstance(gate_mod.g, torch.nn.Parameter):
            with torch.no_grad():
                gate_mod.g.zero_()
        gate_lin = getattr(base_model, 'early_fusion_gate_linear', None)
        if gate_lin is not None:
            with torch.no_grad():
                gate_lin.weight.zero_()
                if gate_lin.bias is not None:
                    gate_lin.bias.zero_()


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", load_video_tower=True, torch_dtype=torch.float16, load_mm_projector=True, zero_early_fusion_gate=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch_dtype

    is_llava = _is_llava_checkpoint(model_path, model_name)

    if is_llava:
        # Load LLaVA model
        is_lora = ('lora' in model_name.lower()) or os.path.exists(os.path.join(model_path, 'adapter_config.json'))
        if is_lora and model_base is None:
            warnings.warn('There is `lora` in model name (or adapter_config.json present) but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if is_lora and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            _sanitize_tower_paths(lora_cfg_pretrained, load_video_tower=load_video_tower)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            embed_dtype = model.get_model().embed_tokens.weight.dtype
            compute_dtype = embed_dtype if embed_dtype.is_floating_point else torch_dtype

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            non_lora_trainables = {k: v.to(compute_dtype) for k, v in non_lora_trainables.items()}
            load_res = model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            if not load_4bit and not load_8bit:
                print('Merging LoRA weights...')
                model = model.merge_and_unload()
            else:
                print('Keeping LoRA adapters unmerged for quantized (4-bit/8-bit) inference...')
            _align_fusion_modules(model, compute_dtype, zero_gate=zero_early_fusion_gate)

            proj_keys = [k for k in non_lora_trainables if 'mm_projector' in k]
            fusion_keys = [k for k in non_lora_trainables if 'early_fusion' in k]
            base_m = model.get_model()
            gate_lin = getattr(base_m, 'early_fusion_gate_linear', None)
            if gate_lin is not None:
                w_norm = gate_lin.weight.float().norm().item()
                b_val = gate_lin.bias.float().item() if gate_lin.bias is not None and gate_lin.bias.numel() == 1 else 0.0
                gate_str = f"gate_v2(w_norm={w_norm:.6f}, bias={b_val:.6f})"
            else:
                gate_str = "N/A"
            print(f"Loaded Stage 2 LoRA + non-LoRA weights from {model_path}: {len(proj_keys)} mm_projector keys, {len(fusion_keys)} early_fusion keys, {gate_str}")
            if zero_early_fusion_gate:
                print("Early-fusion gate zeroed for projector+LoRA ablation")
            if load_res.unexpected_keys:
                print(f"[Info] Unexpected non-LoRA keys ignored: {load_res.unexpected_keys}")
            print('Model is loaded...')
        elif model_base is not None:
            # Stage 1 / adapter-only checkpoint (mm_projector.bin + optional early_fusion_*)
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                _sanitize_tower_paths(cfg_pretrained, load_video_tower=load_video_tower)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                _sanitize_tower_paths(cfg_pretrained, load_video_tower=load_video_tower)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            embed_dtype = model.get_model().embed_tokens.weight.dtype
            compute_dtype = embed_dtype if embed_dtype.is_floating_point else torch_dtype
            mm_proj_path = os.path.join(model_path, 'mm_projector.bin')
            if load_mm_projector and os.path.exists(mm_proj_path):
                mm_projector_weights = torch.load(mm_proj_path, map_location='cpu')
                mm_projector_weights = {k: v.to(compute_dtype) for k, v in mm_projector_weights.items()}
                load_res = model.load_state_dict(mm_projector_weights, strict=False)
                _align_fusion_modules(model, compute_dtype, zero_gate=zero_early_fusion_gate)

                proj_keys = [k for k in mm_projector_weights if 'mm_projector' in k]
                fusion_keys = [k for k in mm_projector_weights if 'early_fusion' in k]
                base_m = model.get_model()
                gate_lin = getattr(base_m, 'early_fusion_gate_linear', None)
                gate_mod = getattr(base_m, 'early_fusion_gate', None)
                if gate_lin is not None:
                    w_norm = gate_lin.weight.float().norm().item()
                    b_val = gate_lin.bias.float().item() if gate_lin.bias is not None and gate_lin.bias.numel() == 1 else 0.0
                    gate_str = f"gate_v2(w_norm={w_norm:.6f}, bias={b_val:.6f})"
                elif gate_mod is not None and hasattr(gate_mod, 'g'):
                    gate_str = f"tanh(g)={torch.tanh(gate_mod.g).item():.6f}"
                else:
                    gate_str = "N/A"
                print(f"Loaded adapter weights from {model_path}: {len(proj_keys)} mm_projector keys, {len(fusion_keys)} early_fusion keys, {gate_str}")
                if zero_early_fusion_gate:
                    print("Early-fusion gate zeroed for projector-only baseline ablation")
                if load_res.unexpected_keys:
                    print(f"[Info] Unexpected adapter keys ignored: {load_res.unexpected_keys}")
            else:
                _align_fusion_modules(model, compute_dtype, zero_gate=True)
                print("No mm_projector.bin loaded — evaluating untrained (step-0) projector with zeroed early-fusion gate")
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                _sanitize_tower_paths(cfg_pretrained, load_video_tower=load_video_tower)
                model, load_info = LlavaLlamaForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, config=cfg_pretrained, output_loading_info=True, **kwargs
                )
                embed_dtype = model.get_model().embed_tokens.weight.dtype
                compute_dtype = embed_dtype if embed_dtype.is_floating_point else torch_dtype
                missing_keys = load_info.get('missing_keys', []) if isinstance(load_info, dict) else []
                no_fusion_in_ckpt = any(
                    'early_fusion_gate_linear' in k or 'early_fusion_gate' in k for k in missing_keys
                )
                _align_fusion_modules(model, compute_dtype, zero_gate=no_fusion_in_ckpt)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    # ==========================================================================================================
    processor = {'image': None, 'video': None}

    if is_llava:
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        tower_dtype = torch_dtype
        if not load_4bit and not load_8bit:
            embed_dtype = model.get_model().embed_tokens.weight.dtype
            if embed_dtype.is_floating_point:
                tower_dtype = embed_dtype

        if getattr(model.config, "mm_image_tower", None) is not None:
            image_tower = model.get_image_tower()
            if not image_tower.is_loaded:
                image_tower.load_model()
            image_tower.to(device=device, dtype=tower_dtype)
            image_processor = image_tower.image_processor
            processor['image'] = image_processor

        if load_video_tower and getattr(model.config, "mm_video_tower", None) is not None:
            video_tower = model.get_video_tower()
            if not video_tower.is_loaded:
                video_tower.load_model()
            video_tower.to(device=device, dtype=tower_dtype)
            video_processor = video_tower.video_processor
            processor['video'] = video_processor
    # ==========================================================================================================

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, processor, context_len
