from dataclasses import dataclass
from typing import Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin
from diffusers.utils import BaseOutput, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)


@dataclass
class FlowMatchDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """Flow Matching Discrete Scheduler."""

    _compatibles = []
    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        reverse: bool = True,
        solver: str = "euler",
    ):
        """Initialize scheduler."""
        self.config_dict = {
            "num_train_timesteps": num_train_timesteps,
            "shift": shift,
            "reverse": reverse,
            "solver": solver,
        }

        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)

        if not reverse:
            sigmas = sigmas.flip(0)

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)
        self._step_index = None

        self.supported_solver = ["euler"]
        if solver not in self.supported_solver:
            raise ValueError(
                f"Solver {solver} not supported. Supported solvers: {self.supported_solver}"
            )

    @property
    def step_index(self):
        return self._step_index

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ):
        """Set timesteps for inference."""
        self.num_inference_steps = num_inference_steps

        sigmas = torch.linspace(1, 0, num_inference_steps + 1)
        sigmas = self.sd3_time_shift(sigmas)

        if not self.config_dict.get("reverse", True):
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.config_dict["num_train_timesteps"]).to(
            dtype=torch.float32, device=device
        )

        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        """Get index for timestep."""
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        return (schedule_timesteps == timestep).nonzero()[0].item()

    def _init_step_index(self, timestep):
        """Initialize step index."""
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        self._step_index = self.index_for_timestep(timestep)

    def sd3_time_shift(self, t: torch.Tensor):
        """Apply SD3 time shift."""
        shift = self.config_dict.get("shift", 1.0)
        return (shift * t) / (1 + (shift - 1) * t)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[FlowMatchDiscreteSchedulerOutput, Tuple]:
        """Execute one step of the scheduler."""

        if (
            isinstance(timestep, int) or
            isinstance(timestep, torch.IntTensor) or
            isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                "Passing integer indices as timesteps is not supported. "
                "Pass one of the `scheduler.timesteps` as a timestep."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)
        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]

        solver = self.config_dict.get("solver", "euler")
        if solver == "euler":
            prev_sample = sample + model_output.to(torch.float32) * dt
        else:
            raise ValueError(f"Solver {solver} not supported")

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchDiscreteSchedulerOutput(prev_sample=prev_sample)


def get_scheduler(cfg):
    """Instantiate scheduler from config."""
    return FlowMatchDiscreteScheduler(
        num_train_timesteps=getattr(cfg, 'num_train_timesteps', 1000),
        shift=getattr(cfg, 'shift', 1.0),
        reverse=getattr(cfg, 'reverse', True),
        solver=getattr(cfg, 'solver', 'euler'),
    )
