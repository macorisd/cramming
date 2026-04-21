"""Interface to construct models."""

from .huggingface_interface import construct_huggingface_model
from .funnel_transformers import construct_scriptable_funnel
from .recurrent_transformers import construct_scriptable_recurrent
from .sanity_check import SanityCheckforPreTraining
from .crammed_bert import construct_crammed_bert

import logging
from ..utils import is_main_process

log = logging.getLogger(__name__)


def construct_model(cfg_arch, vocab_size, downstream_classes=None):
    model = None
    embedding_cfg = cfg_arch.get("embedding", None)
    if embedding_cfg is not None and embedding_cfg.get("pos_embedding") in {"sinusoidal", "scaled-sinusoidal"} and is_main_process():
        positional_wave = embedding_cfg.get("positional_wave", "sinusoid")
        log.info(f"Using absolute positional encoding '{embedding_cfg.pos_embedding}' with wave '{positional_wave}'.")
    if cfg_arch.architectures is not None:
        # attempt to solve locally
        if "ScriptableCrammedBERT" in cfg_arch.architectures:
            model = construct_crammed_bert(cfg_arch, vocab_size, downstream_classes)
        elif "ScriptableFunnelLM" in cfg_arch.architectures:
            model = construct_scriptable_funnel(cfg_arch, vocab_size, downstream_classes)
        elif "ScriptableRecurrentLM" in cfg_arch.architectures:
            model = construct_scriptable_recurrent(cfg_arch, vocab_size, downstream_classes)
        elif "SanityCheckLM" in cfg_arch.architectures:
            model = SanityCheckforPreTraining(cfg_arch.width, vocab_size)

    if model is not None:  # Return local model arch
        num_params = sum([p.numel() for p in model.parameters()])
        if is_main_process():
            log.info(f"Model with architecture {cfg_arch.architectures[0]} loaded with {num_params:,} parameters.")
        return model

    try:  # else try on HF
        model = construct_huggingface_model(cfg_arch, vocab_size, downstream_classes)
        num_params = sum([p.numel() for p in model.parameters()])
        if is_main_process():
            log.info(f"Model with config {cfg_arch} loaded with {num_params:,} parameters.")
        return model
    except Exception as e:
        raise ValueError(f"Invalid model architecture {cfg_arch.architectures} given. Error: {e}")
