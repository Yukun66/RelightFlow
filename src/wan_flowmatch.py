from dataclasses import dataclass
from typing import Tuple, Union
import torch

from diffusers.utils import BaseOutput, logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor

def wan_flowmatch_light(
    self,
    V,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    return_dict: bool = True,
) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:

    if self.step_index is None:
        self._init_step_index(timestep)

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)

    sigma = self.sigmas[self.step_index]
    sigma_next = self.sigmas[self.step_index + 1]
    
    prev_sample = sample + (sigma_next - sigma) * V
    
    # Cast sample back to model compatible dtype
    prev_sample = prev_sample.to(model_output.dtype)

    # upon completion increase step index by one
    self._step_index += 1

    if not return_dict:
        return (prev_sample,)

    return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)