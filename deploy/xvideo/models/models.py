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

    dtype = PRECISION_TO_TYPE[cfg.dit_precision]
    # Create the model on CPU first. Streaming the checkpoint below loads tensors
    # one-by-one directly into the model, keeping peak memory at ~1x model size.
    # Move to GPU after loading if device != CPU.
    model = Transformer3DModel(
        dtype=dtype, device=torch.device("cpu"), **_arch_params(cfg.dit_arch_config)
    )
    # Convert model to target dtype (fixes bias dtype)
    model = model.to(dtype=dtype)
    logger.info(f"Empty model created on CPU: {torch.cuda.memory_allocated()/1e9:.1f}GB GPU allocated")

    if state_dict is not None:
        for prefix in ("model.", "module.", "transformer."):
            if any(k.startswith(prefix) for k in state_dict):
                state_dict = {
                    (k[len(prefix):] if k.startswith(prefix) else k): v
                    for k, v in state_dict.items()
                }

        # ONE streaming pass: dequantize each checkpoint tensor and copy it straight
        # into the model's (already GPU-resident) param, then free it. No intermediate
        # load_state_dict dict -> peak stays ~1x model size instead of 2x (which OOM'd).
        targets = dict(model.named_parameters())
        targets.update(dict(model.named_buffers()))
        n = len(state_dict)
        logger.info(f"Streaming {n} tensors into the GPU model ({dtype})...")
        loaded, unexpected = 0, []
        for i, (k, v) in enumerate(state_dict.items()):
            target = targets.get(k)
            if target is None:
                unexpected.append(k)
                continue
            # dequantize int8 {data,scale}, else take the raw tensor
            if isinstance(v, dict) and "data" in v:
                v = v["data"].float() * v["scale"]
            elif not isinstance(v, torch.Tensor):
                continue
            v = v.to(device=device, dtype=dtype)  # move ONE tensor to target dtype/device
            if k == "img_in.weight" and target.shape != v.shape:
                logger.info(f"Inflate {k} from {v.shape} to {target.shape}")
                v = v.reshape_as(v.new_zeros(target.shape))
            with torch.no_grad():
                # Ensure target is in correct dtype before copy
                if target.dtype != dtype:
                    target.data = target.data.to(dtype=dtype)
                target.copy_(v)
            del v
            loaded += 1
            if i % 100 == 0:
                logger.info(f"  [{i}/{n}] streamed | GPU allocated: {torch.cuda.memory_allocated()/1e9:.1f}GB")

        missing = [k for k in targets if k not in state_dict]
        del state_dict
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Loaded {loaded}/{n} tensors. GPU allocated: {torch.cuda.memory_allocated()/1e9:.1f}GB")
        if missing:
            logger.warning(f"Missing keys when loading DiT: {missing[:20]}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading DiT: {unexpected[:20]}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Instantiate model with {total_params / 1e9:.2f}B parameters")
    logger.info(f"DiT on {device}: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated")

    return model.eval()


__all__ = [
    "load_dit",
    "load_pipeline",
    "build_vae",
]
