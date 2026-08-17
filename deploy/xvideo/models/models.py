import torch
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration
from loguru import logger

from xvideo.models.dit import Transformer3DModel
from xvideo.models.vae import XVAEChunkCausal
from xvideo.models.scheduler import FlowMatchDiscreteScheduler
from xvideo.models.pipeline import Pipeline, PRECISION_TO_TYPE


def load_text_encoder(
    text_encoder_ckpt: str,
    device: torch.device = torch.device("cpu"),
    torch_dtype: torch.dtype = torch.bfloat16,
):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        text_encoder_ckpt,
        dtype=torch_dtype,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    tokenizer = Qwen2Tokenizer.from_pretrained(
        text_encoder_ckpt,
        local_files_only=True,
    )
    return tokenizer, model


def _arch_params(arch_config) -> dict:
    return dict(arch_config.get("params", {})) if isinstance(arch_config, dict) else {}


def build_vae(cfg, device: torch.device):
    pretrained = cfg.vae_arch_config["pretrained"]
    vae = XVAEChunkCausal.from_pretrained(
        pretrained,
        torch_dtype=PRECISION_TO_TYPE[cfg.vae_precision],
        device=device,
        **_arch_params(cfg.vae_arch_config),
    )
    return vae


def load_pipeline(cfg, dit, device: torch.device):
    vae = build_vae(cfg, device)

    tokenizer, text_encoder = load_text_encoder(
        torch_dtype=PRECISION_TO_TYPE[cfg.text_encoder_precision],
        device=device,
        **_arch_params(cfg.text_encoder_arch_config),
    )

    scheduler = FlowMatchDiscreteScheduler(**_arch_params(cfg.scheduler_arch_config))

    pipeline = Pipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=dit,
        scheduler=scheduler,
        args=cfg,
        **_arch_params(cfg.pipeline_arch_config),
    )
    return pipeline.to(device)


def load_dit(cfg, device: torch.device) -> torch.nn.Module:
    state_dict = None
    if cfg.dit_ckpt is not None:
        logger.info(f"Loading DiT checkpoint: {cfg.dit_ckpt}")
        # mmap=True memory-maps the (32+ GB) checkpoint instead of reading it all into
        # CPU RAM, which OOM-kills (SIGKILL) on shared boxes. mmap requires
        # map_location='cpu'; load_state_dict below copies the mapped tensors straight
        # into the GPU model, so peak CPU RAM stays tiny.
        state_dict = torch.load(cfg.dit_ckpt, map_location="cpu", weights_only=True, mmap=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]

    # Load model in fp32 to avoid dtype mismatch during initialization,
    # then convert to target dtype after all internal tensors are created
    model = Transformer3DModel(
        dtype=torch.float32, device=device, **_arch_params(cfg.dit_arch_config)
    )

    if state_dict is not None:
        for prefix in ("model.", "module.", "transformer."):
            if any(k.startswith(prefix) for k in state_dict):
                state_dict = {
                    (k[len(prefix):] if k.startswith(prefix) else k): v
                    for k, v in state_dict.items()
                }

        load_state_dict = {}
        for k, v in state_dict.items():
            # Keep checkpoint tensors on CPU (mmap'd). PyTorch will transfer to GPU
            # during load_state_dict. Converting dtype here is OK (CPU is fast).
            if isinstance(v, torch.Tensor):
                v = v.to(dtype=torch.float32)

            if (
                k == "img_in.weight" and
                hasattr(model, "img_in") and
                model.img_in.weight.shape != v.shape
            ):
                logger.info(f"Inflate {k} from {v.shape} to {model.img_in.weight.shape}")
                v = v.reshape_as(v.new_zeros(model.img_in.weight.shape))
            load_state_dict[k] = v

        missing_keys, unexpected_keys = model.load_state_dict(load_state_dict, strict=True)
        if missing_keys:
            logger.warning(f"Missing keys when loading DiT: {missing_keys[:20]}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading DiT: {unexpected_keys[:20]}")

        # Free checkpoint memory before moving model to device
        del load_state_dict
        del state_dict
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Instantiate model with {total_params / 1e9:.2f}B parameters")

    # Move to device and convert to target dtype (bf16 for joyomni_ops, fp16 otherwise)
    target_dtype = PRECISION_TO_TYPE[cfg.dit_precision]
    model = model.to(device=device, dtype=target_dtype)

    return model.eval()


__all__ = [
    "load_dit",
    "load_pipeline",
    "build_vae",
]
