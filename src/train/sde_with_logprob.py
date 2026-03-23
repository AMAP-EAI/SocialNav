import torch
import math
from typing import Optional, Union, Tuple, List
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

class ConditionalFlowMatcherWithSigmaSchedule(FlowMatchEulerDiscreteScheduler):
    def __init__(self, num_inference_steps=3, noise_level=0.1, device="cpu"):
        super().__init__(num_train_timesteps=num_inference_steps)
        self.num_inference_steps = num_inference_steps
        self.noise_level = noise_level
        self.timesteps = np.linspace(1.0, 0, self.num_inference_steps+1)
        self.timesteps = torch.from_numpy(self.timesteps).to(device)

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = torch.where(torch.isclose(schedule_timesteps, timestep, atol=1e-6))[0]

        if len(indices) == 0:
            raise ValueError(f"No timestep in schedule is close to {timestep} within tolerance.")

        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()
    def sde_step_with_logprob(
        self,
        v_t: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        x_t: torch.FloatTensor,
        # noise_level: float = 0.7,
        x_t_new: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
        process from the learned model outputs (most often the predicted velocity).

        Args:
            v_t (`torch.FloatTensor`):
                The direct output from learned flow model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            x_t (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
        """
        # Use fp32 for SDE mean; bf16 can overflow.
        v_t = -1 * v_t.float()
        x_t = x_t.float()
        if x_t_new is not None:
            x_t_new=x_t_new.float()

        timestep = timestep.to(self.timesteps.device).to(self.timesteps.dtype)
        step_index = [self.index_for_timestep(t) for t in timestep]
        prev_step_index = [step+1 for step in step_index]

        t = self.timesteps[step_index].view(-1, *([1] * (len(x_t.shape) - 1)))
        t_prev = self.timesteps[prev_step_index].view(-1, *([1] * (len(x_t.shape) - 1)))
        t_max = self.timesteps[1].item()
        dt = abs(t_prev - t)

        sigma_t = torch.sqrt(t / (1 - torch.where(t == 1, t_max, t))) * self.noise_level

        x_t_mean = x_t*(1+sigma_t**2/(2*t)*(-1*dt)) + v_t*(1+sigma_t**2*(1-t)/(2*t))*(-1*dt)
        
        if x_t_new is None:
            variance_noise = randn_tensor(
                v_t.shape,
                generator=generator,
                device=v_t.device,
                dtype=v_t.dtype,
            )
            x_t_new = x_t_mean + sigma_t * torch.sqrt(dt) * variance_noise

        log_prob = (
            -((x_t_new.detach() - x_t_mean) ** 2) / (2 * ((sigma_t * torch.sqrt(dt))**2))
            - torch.log(sigma_t * torch.sqrt(dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        
        return x_t_new, log_prob, x_t_mean, sigma_t
 
class ConditionalFlowMatcherWithSigmaSchedule_CPS(FlowMatchEulerDiscreteScheduler):
    def __init__(self, num_inference_steps=3, noise_level=0.1, device="cpu"):
        super().__init__(num_train_timesteps=num_inference_steps)
        self.num_inference_steps = num_inference_steps
        self.noise_level = noise_level
        self.timesteps = np.linspace(1.0, 0, self.num_inference_steps+1)
        self.timesteps = torch.from_numpy(self.timesteps).to(device)

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = torch.where(torch.isclose(schedule_timesteps, timestep, atol=1e-6))[0]

        if len(indices) == 0:
            raise ValueError(f"No timestep in schedule is close to {timestep} within tolerance.")

        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()
    def sde_step_with_logprob(
        self,
        v_t: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        x_t: torch.FloatTensor,
        # noise_level: float = 0.7,
        x_t_new: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
        process from the learned model outputs (most often the predicted velocity).

        Args:
            v_t (`torch.FloatTensor`):
                The direct output from learned flow model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            x_t (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
        """
        v_t = -1 * v_t.float()
        x_t = x_t.float()
        if x_t_new is not None:
            x_t_new=x_t_new.float()

        timestep = timestep.to(self.timesteps.device).to(self.timesteps.dtype)
        step_index = [self.index_for_timestep(t) for t in timestep]
        prev_step_index = [step+1 for step in step_index]

        t = self.timesteps[step_index].view(-1, *([1] * (len(x_t.shape) - 1)))
        t_prev = self.timesteps[prev_step_index].view(-1, *([1] * (len(x_t.shape) - 1)))
        dt = abs(t_prev - t)

        sigma_t = t_prev * math.sin(self.noise_level * math.pi / 2)

        pred_original_sample = x_t - t * v_t
        noise_estimate = x_t + v_t * (1 - t)
        x_t_mean = pred_original_sample * (1 - t_prev) + noise_estimate * torch.sqrt(t_prev**2 - sigma_t**2)

        if x_t_new is None:
            variance_noise = randn_tensor(
                v_t.shape,
                generator=generator,
                device=v_t.device,
                dtype=v_t.dtype,
            )
            x_t_new = x_t_mean + sigma_t * variance_noise

        log_prob = -((x_t_new.detach() - x_t_mean) ** 2)

        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        
        return x_t_new, log_prob, x_t_mean, sigma_t
