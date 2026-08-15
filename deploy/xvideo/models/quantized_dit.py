"""
Quantized DiT Model - accepts qint8 weights directly without conversion.
Wraps Transformer3DModel to handle quantized checkpoint loading.
"""

import torch
import torch.nn as nn
from loguru import logger
from .dit import Transformer3DModel


class QuantizedTransformer3DModel(Transformer3DModel):
    """
    Quantized-aware wrapper for Transformer3DModel.

    Loads qint8 checkpoint directly without type conversion.
    Replaces float32 parameters with quantized buffers.
    """

    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict with qint8 tensor support.

        Quantized tensors replace float parameters in-place.
        """
        quantized_count = 0
        float_count = 0
        replaced_params = {}

        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and v.is_quantized:
                quantized_count += 1
                # Find the target module and parameter name
                if "." in k:
                    module_path, param_name = k.rsplit(".", 1)
                    module = self
                    try:
                        for part in module_path.split("."):
                            module = getattr(module, part)
                        # Check if parameter exists and remove it
                        if hasattr(module, param_name):
                            delattr(module, param_name)
                        # Register as quantized buffer
                        module.register_buffer(param_name, v, persistent=True)
                        replaced_params[k] = "buffer"
                    except (AttributeError, KeyError) as e:
                        logger.warning(f"Could not process {k}: {e}")
                else:
                    # Top-level parameter
                    if hasattr(self, k):
                        delattr(self, k)
                    self.register_buffer(k, v, persistent=True)
                    replaced_params[k] = "buffer"
            else:
                float_count += 1

        if quantized_count > 0:
            logger.info(f"Replaced {quantized_count} parameters with quantized buffers, kept {float_count} float tensors")
            return [], []

        # If no quantized tensors, use parent's load_state_dict
        return super().load_state_dict(state_dict, strict=strict)

    def to(self, *args, **kwargs):
        """
        Custom device movement to avoid OOM when moving all quantized buffers at once.
        Moves quantized buffers incrementally.
        """
        # Extract device from args/kwargs
        device = None
        if args:
            arg = args[0]
            if isinstance(arg, torch.device):
                device = arg
            elif isinstance(arg, str):
                device = torch.device(arg)
        elif 'device' in kwargs:
            device = kwargs['device']

        if device is None:
            return super().to(*args, **kwargs)

        # Move float parameters normally
        for name, param in self.named_parameters():
            if param is not None:
                param.data = param.data.to(device)

        # Move quantized buffers incrementally to avoid OOM
        quantized_buffers = [(name, buf) for name, buf in self.named_buffers()
                            if buf is not None and buf.is_quantized]
        if quantized_buffers:
            logger.info(f"Moving {len(quantized_buffers)} quantized buffers to {device} incrementally...")
            for i, (name, buf) in enumerate(quantized_buffers):
                # Move one buffer at a time
                moved_buf = buf.to(device)
                # Update buffer in-place
                self._buffers[name] = moved_buf
                if (i + 1) % 100 == 0:
                    logger.info(f"  Moved {i + 1}/{len(quantized_buffers)} buffers")

        # Move float buffers normally
        for name, buf in self.named_buffers():
            if buf is not None and not buf.is_quantized:
                self._buffers[name] = buf.to(device)

        return self

    def forward(self, x, t, context, rotary_emb=None, image_rotary_emb=None):
        """
        Forward pass with automatic dequantization of qint8 weights.
        Converts quantized buffers to float32 for PyTorch compatibility.
        """
        # Dequantize all qint8 buffers before forward
        for name, buf in list(self.named_buffers()):
            if buf is not None and buf.is_quantized:
                dequantized = torch.dequantize(buf)
                self._buffers[name] = dequantized

        return super().forward(x, t, context, rotary_emb, image_rotary_emb)
