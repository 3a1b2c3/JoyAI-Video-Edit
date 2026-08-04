from __future__ import annotations

import functools

import torch
import torch.nn as nn


_configured: set[int] = set()
_configured_encode: set[int] = set()
_configured_encode_dynamic: set[int] = set()


def _original_callable(module, name: str):
    original_name = f"_vae_compile_original{name}"
    original = getattr(module, original_name, None)
    if original is not None:
        return original

    original = getattr(module, name)
    while hasattr(original, "_torchdynamo_orig_callable"):
        original = original._torchdynamo_orig_callable
    setattr(module, original_name, original)
    return original


def _is_fx_tracing() -> bool:
    symbolic_trace = torch.fx._symbolic_trace
    check = getattr(symbolic_trace, "is_fx_symbolic_tracing", None)
    if check is None:
        check = symbolic_trace.is_fx_tracing
    return bool(check())


def _fx_safe_compiled(compiled, fallback):
    @functools.wraps(fallback)
    def wrapped(*args, **kwargs):
        if _is_fx_tracing():
            return fallback(*args, **kwargs)
        return compiled(*args, **kwargs)

    return wrapped


def maybe_setup_decode(vae) -> None:
    if id(vae) in _configured:
        return
    n_conv = 0
    for m in vae.modules():
        if isinstance(m, nn.Conv3d):
            m.weight.data = m.weight.data.to(memory_format=torch.channels_last_3d)
            n_conv += 1
    if hasattr(vae, "_decode"):
        original = _original_callable(vae, "_decode")
        compiled = torch.compile(original, mode="max-autotune-no-cudagraphs", dynamic=False)
        vae._decode = _fx_safe_compiled(compiled, original)
        target = "_decode"
    elif hasattr(vae, "decode"):
        original = _original_callable(vae, "decode")
        compiled = torch.compile(original, mode="max-autotune-no-cudagraphs", dynamic=False)
        vae.decode = _fx_safe_compiled(compiled, original)
        target = "decode"
    else:
        raise RuntimeError("VAE has neither _decode nor decode; cannot compile")
    _configured.add(id(vae))
    print(f"[vae_compile] converted {n_conv} Conv3d weights to channels_last_3d + compiled vae.{target}")


def prep_input(z: torch.Tensor) -> torch.Tensor:
    return z.to(memory_format=torch.channels_last_3d)


def maybe_setup_encode(vae) -> None:
    if id(vae) in _configured_encode:
        return
    n_conv = 0
    for m in vae.modules():
        if isinstance(m, nn.Conv3d):
            m.weight.data = m.weight.data.to(memory_format=torch.channels_last_3d)
            n_conv += 1
    if hasattr(vae, "_encode"):
        original = _original_callable(vae, "_encode")
        compiled = torch.compile(original, mode="max-autotune-no-cudagraphs", dynamic=False)
        vae._encode = _fx_safe_compiled(compiled, original)
        target = "_encode"
    elif hasattr(vae, "encode"):
        original = _original_callable(vae, "encode")
        compiled = torch.compile(original, mode="max-autotune-no-cudagraphs", dynamic=False)
        vae.encode = _fx_safe_compiled(compiled, original)
        target = "encode"
    else:
        raise RuntimeError("VAE has neither _encode nor encode; cannot compile")
    _configured_encode.add(id(vae))
    print(f"[vae_compile] converted {n_conv} Conv3d weights to channels_last_3d + compiled vae.{target} (encode)")


def warmup_encode(vae, in_channels: int, h_px: int, w_px: int,
                  device: torch.device, dtype: torch.dtype,
                  temporal_lens: tuple[int, ...] = (1, 9),
                  autocast: bool = False) -> None:
    maybe_setup_encode(vae)
    from contextlib import nullcontext
    dev_type = torch.device(device).type
    use_ac = autocast and dev_type in {"cuda", "cpu"}
    for t in temporal_lens:
        x = torch.zeros(1, in_channels, t, h_px, w_px, device=device, dtype=dtype)
        x = prep_input(x)
        ctx = (
            torch.autocast(device_type=dev_type, dtype=dtype, enabled=True)
            if use_ac else nullcontext()
        )
        try:
            with torch.no_grad(), ctx:
                _ = vae.encode(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            print(f"[vae_compile] warmup compiled encode shape (1,{in_channels},{t},{h_px},{w_px}) autocast={autocast}")
        except Exception as exc:  # noqa: BLE001
            print(f"[vae_compile] encode warmup failed for t={t}: {exc!r}")


def maybe_setup_encode_dynamic(vae) -> None:
    if id(vae) in _configured_encode_dynamic:
        return
    if hasattr(vae, "_encode"):
        dynamic_core = _original_callable(vae, "_encode")
    elif hasattr(vae, "encode"):
        dynamic_core = _original_callable(vae, "encode")
    else:
        raise RuntimeError("VAE has neither _encode nor encode; cannot compile")
    compiled = torch.compile(dynamic_core, mode="max-autotune-no-cudagraphs", dynamic=True)
    vae._encode_dynamic = _fx_safe_compiled(compiled, dynamic_core)
    _configured_encode_dynamic.add(id(vae))
    print("[vae_compile] compiled vae._encode_dynamic (dynamic=True, reference-image path)")


def encode_via_dynamic(vae, x: torch.Tensor):
    fn = getattr(vae, "_encode_dynamic", None)
    if fn is None:
        return vae.encode(x)
    from xvideo.models.vae.vae import (
        DiagonalGaussianDistribution,
        EncoderOutput,
    )
    h = fn(prep_input(x))
    return EncoderOutput(latent_dist=DiagonalGaussianDistribution(h))


def warmup_encode_dynamic(vae, in_channels: int, hw_list, device: torch.device,
                          dtype: torch.dtype, temporal_lens: tuple[int, ...] = (1,),
                          autocast: bool = False) -> None:
    fn = getattr(vae, "_encode_dynamic", None)
    if fn is None:
        print("[vae_compile] warmup_encode_dynamic skipped: _encode_dynamic not set up")
        return
    from contextlib import nullcontext
    dev_type = torch.device(device).type
    use_ac = autocast and dev_type in {"cuda", "cpu"}
    n_ok = 0
    for (h_px, w_px) in hw_list:
        for t in temporal_lens:
            x = torch.zeros(1, in_channels, t, h_px, w_px, device=device, dtype=dtype)
            x = prep_input(x)
            ctx = (
                torch.autocast(device_type=dev_type, dtype=dtype, enabled=True)
                if use_ac else nullcontext()
            )
            try:
                with torch.no_grad(), ctx:
                    _ = fn(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[vae_compile] dynamic encode warmup failed for ({h_px},{w_px},t={t}): {exc!r}")
    print(f"[vae_compile] dynamic encode warmup done: {n_ok}/{len(hw_list) * len(temporal_lens)} shapes autocast={autocast}")


def warmup_decode(vae, latent_channels: int, h_lat: int, w_lat: int,
                  device: torch.device, dtype: torch.dtype,
                  temporal_lens: tuple[int, ...] = (1, 2),
                  autocast: bool = True) -> None:
    maybe_setup_decode(vae)
    from contextlib import nullcontext
    dev_type = torch.device(device).type
    use_ac = autocast and dev_type in {"cuda", "cpu"}
    for t in temporal_lens:
        z = torch.zeros(1, latent_channels, t, h_lat, w_lat, device=device, dtype=dtype)
        z = prep_input(z)
        ctx = (
            torch.autocast(device_type=dev_type, dtype=dtype, enabled=True)
            if use_ac else nullcontext()
        )
        try:
            with torch.no_grad(), ctx:
                _ = vae.decode(z, return_dict=False)[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            print(f"[vae_compile] warmup compiled decode shape (1,{latent_channels},{t},{h_lat},{w_lat}) autocast={autocast}")
        except Exception as exc:  # noqa: BLE001
            print(f"[vae_compile] warmup failed for t={t}: {exc!r}")
