import copy
import os
import torch

from vitra.utils.hf_env import enable_hf_offline

enable_hf_offline()
import transformers

def build_vlm(vlm_config):
    vlm_config = copy.deepcopy(vlm_config)
    model_path = vlm_config.get("pretrained_model_name_or_path")
    model_name = vlm_config.get("name")
    model_type = vlm_config.get("type", "AutoModel")
    if model_name == "paligemma":
        from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            local_files_only=True,
            # attn_implementation="eager",
            # revision="bfloat16",
        )
        processor = PaliGemmaProcessor.from_pretrained(model_path, local_files_only=True)
    elif model_name in ("llava_ov2", "llava_onevision2"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        torch_dtype = torch.bfloat16 if vlm_config.get("use_bf16", True) else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="cpu",
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    return processor, model
