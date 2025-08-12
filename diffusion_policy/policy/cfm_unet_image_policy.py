from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

import numpy as np
from diffusion_policy.sde_lib import ConsistencyFM

class CFMUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            Conditional_ConsistencyFM=None, 
            eta=0.01,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if Conditional_ConsistencyFM is None:
                    Conditional_ConsistencyFM = {
                        'eps': 1e-2,
                        'num_segments': 1,
                        'boundary': 1,
                        'delta': 1e-2,
                        'alpha': 1e-5,
                        'num_inference_step': 1
                    }
        self.eta = eta
        self.eps = Conditional_ConsistencyFM['eps']
        self.num_segments = Conditional_ConsistencyFM['num_segments']
        self.boundary = Conditional_ConsistencyFM['boundary']
        self.delta = Conditional_ConsistencyFM['delta']
        self.alpha = Conditional_ConsistencyFM['alpha']
        self.num_inference_step = Conditional_ConsistencyFM['num_inference_step']
        
        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    # # ========= inference  ============
    # def conditional_sample(self, 
    #         condition_data, condition_mask,
    #         local_cond=None, global_cond=None,
    #         generator=None,
    #         # keyword arguments to scheduler.step
    #         **kwargs
    #         ):

    #     trajectory = torch.randn(
    #         size=condition_data.shape, 
    #         dtype=condition_data.dtype,
    #         device=condition_data.device,
    #         generator=generator)
    
    #     # set step values
    #     scheduler.set_timesteps(self.num_inference_steps)

    #     for t in scheduler.timesteps:
    #         # 1. apply conditioning
    #         trajectory[condition_mask] = condition_data[condition_mask]

    #         # 2. predict model output
    #         model_output = self.model(trajectory, t, 
    #             local_cond=local_cond, global_cond=global_cond)

    #         # 3. compute previous image: x_t -> x_t-1
    #         trajectory = scheduler.step(
    #             model_output, t, trajectory, 
    #             generator=generator,
    #             **kwargs
    #             ).prev_sample
        
    #     # finally make sure conditioning is enforced
    #     trajectory[condition_mask] = condition_data[condition_mask]        

    #     return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        noise = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype,
            device=cond_data.device,
            generator=None)
        z = noise.detach().clone() # a0

        sde = ConsistencyFM('gaussian', 
                            noise_scale=1.0,  
                            use_ode_sampler='rk45', # unused
                            sigma_var=0.0, 
                            ode_tol=1e-5, 
                            sample_N= self.num_inference_step)

        # Uniform
        dt = 1./self.num_inference_step
        eps = self.eps

        for i in range(sde.sample_N):
            num_t = i /sde.sample_N * (1 - eps) + eps
            t = torch.ones(z.shape[0], device=noise.device) * num_t
            pred = self.model(z, t*99, local_cond=local_cond, global_cond=global_cond) ### Copy from models/utils.py 
            # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability 
            sigma_t = sde.sigma_t(num_t)
            pred_sigma = pred + (sigma_t**2)/(2*(sde.noise_scale**2)*((1.-num_t)**2)) * (0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*z.detach().clone())
            z = z.detach().clone() + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)
        z[cond_mask] = cond_data[cond_mask] # a1
        # unnormalize prediction
        naction_pred = z[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        eps = self.eps
        num_segments = self.num_segments
        boundary = self.boundary
        delta  = self.delta
        alpha =  self.alpha
        reduce_op = torch.mean
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        target = nactions
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        # gt & noise
        target = target
        a0 =  torch.randn(trajectory.shape, device=trajectory.device)
       
        t = torch.rand(target.shape[0], device=target.device) * (1 - eps) + eps # 1=sde.T
        r = torch.clamp(t + delta, max=1.0)
        t_expand = t.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        r_expand = r.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        xt = t_expand * target + (1.-t_expand) * a0
        xr = r_expand * target + (1.-r_expand) * a0
        #apply mask
        xt[condition_mask] = cond_data[condition_mask]
        xr[condition_mask] = cond_data[condition_mask]

        segments = torch.linspace(0, 1, num_segments + 1, device=target.device)
        seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1) # .clamp(min=1) prevents the inclusion of 0 in indices.
        segment_ends = segments[seg_indices]
        segment_ends_expand = segment_ends.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        x_at_segment_ends = segment_ends_expand * target + (1.-segment_ends_expand) * a0
    
        def f_euler(t_expand, segment_ends_expand, xt, vt):
            return xt + (segment_ends_expand - t_expand) * vt
        def threshold_based_f_euler(t_expand, segment_ends_expand, xt, vt, threshold, x_at_segment_ends):
            if (threshold, int) and threshold == 0:
                return x_at_segment_ends
      
            less_than_threshold = t_expand < threshold
      
            res = (
        less_than_threshold * f_euler(t_expand, segment_ends_expand, xt, vt)
        + (~less_than_threshold) * x_at_segment_ends
        )
            return res
        vt = self.model(xt, t*99, cond=local_cond, global_cond=global_cond)
        vr = self.model(xr, r*99, local_cond=local_cond, global_cond=global_cond)
        # mask
        vt[condition_mask] = cond_data[condition_mask]
        vr[condition_mask] = cond_data[condition_mask]

        vr = torch.nan_to_num(vr)
      
        ft = f_euler(t_expand, segment_ends_expand, xt, vt)
        fr = threshold_based_f_euler(r_expand, segment_ends_expand, xr, vr, boundary, x_at_segment_ends)

        ##### loss #####
        losses_f = torch.square(ft - fr)
        losses_f = reduce_op(losses_f.reshape(losses_f.shape[0], -1), dim=-1)
    
        def masked_losses_v(vt, vr, threshold, segment_ends, t):
            if (threshold, int) and threshold == 0:
                return 0
    
            less_than_threshold = t_expand < threshold
      
            far_from_segment_ends = (segment_ends - t) > 1.01 * delta
            far_from_segment_ends = far_from_segment_ends.view(-1, 1, 1).repeat(1, trajectory.shape[1], trajectory.shape[2])
      
            losses_v = torch.square(vt - vr)
            losses_v = less_than_threshold * far_from_segment_ends * losses_v
            losses_v = reduce_op(losses_v.reshape(losses_v.shape[0], -1), dim=-1)
      
            return losses_v
    
        losses_v = masked_losses_v(vt, vr, boundary, segment_ends, t)

        loss = torch.mean(losses_f + alpha * losses_v)
        # loss_dict = { 'bc_loss': 
        #              loss.item(),}
        
        # return loss, loss_dict
        return loss
