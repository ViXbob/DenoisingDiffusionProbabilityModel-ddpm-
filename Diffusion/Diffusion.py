
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
import os
from typing import List
from pathlib import Path

import math

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1)).contiguous()

def normal_kl(mean1, logvar1, mean2, logvar2):
  """
  KL divergence between normal distributions parameterized by mean and log-variance.
  """
  return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + F.mse_loss(mean1, mean2, reduction = 'none') * torch.exp(-logvar2))

def approx_standard_normal_cdf(x):
    return 0.5 * ( 1.0 + torch.tanh( math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3)) ) )

def discretized_gaussian_log_likelihood(x, means, log_scales):
    # Assumes data is integers [0, 255] rescaled to [-1, 1]
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(torch.clamp(cdf_plus, min=1e-12))
    log_one_minus_cdf_min = torch.log(torch.clamp(1. - cdf_min, min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min,
                torch.log(torch.clamp(cdf_delta, min=1e-12))))
    assert log_probs.shape == x.shape
    return log_probs

class GaussianDiffusion(nn.Module):
    def __init__(self, model, betas, T, model_mean_type: str, model_var_type: str, loss_type: str):
        """
        model_mean_type \in ["xprev", "xstart", "eps"]
            1. xprev: the model predict x_{t - 1}
            2. xstart: the model predict x_0
            3. eps: the model predict epsilon
        model_var_type \in ["learned", "fiexedsmall", "fixedlarge"]
            1. learned: model will predict variance
            2. fixedsmall: var_t is equal to variance of q(x_{t-1} | x_t, x_0)
            3. fixedlarge: var_t = beta_t * I
        loss_type \in ["simple", "KL"]
            1. simple: L_simple
            2. KL: Use KL divergence as loss
        """
        super().__init__()
        assert isinstance(betas, torch.Tensor)
        assert (betas > 0).all() and (betas <= 1).all()
        assert T == len(betas)
        self.T = T
        self.mean_type = model_mean_type
        self.var_type = model_var_type
        self.loss_type = loss_type
        
        self.register_buffer('betas', betas.double())
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_bar', torch.cumprod(self.alphas, dim = 0))
        self.register_buffer('alphas_bar_prev', F.pad(self.alphas_bar, [1, 0], value=1)[:T])
        
        assert self.alphas_bar_prev.shape[0] == T
        
        self.model = model
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(self.alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - self.alphas_bar))
        self.register_buffer('sqrt_inv_alphas_bar', torch.sqrt(1. / self.alphas_bar))
        self.register_buffer('sqrt_inv_alphas_bar_m1', torch.sqrt(1. / self.alphas_bar - 1.))
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_variance', betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)) # variance diagonal
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.cat((self.posterior_variance[1:2], self.posterior_variance[1:]), dim = 0)))
        # mean of q(x_{t - 1} | x_t, x_0) = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(self.alphas_bar_prev) / (1. - self.alphas_bar))
        self.register_buffer('posterior_mean_coef2', (1. - self.alphas_bar_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_bar))
        
        self.register_buffer('log_betas', torch.log(torch.cat((self.posterior_variance[1:2], self.betas[1:]), dim = 0)))
        
        self.register_buffer('coeff1', torch.sqrt(1. / self.alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - self.alphas) / torch.sqrt(1. - self.alphas_bar))

    def q_sample(self, x_0, t, noise = None):
        """
        Sample x_t condition on x_0
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        assert noise.shape == x_0.shape
        
        # x_t = \sqrt{\bar\alpha_t} * x_0 + \sqrt{1 - \bar\alpha_t} * noise
        return (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        
    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute q(x_{t - 1} | x_t, x_0) mean and variance
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] == x_0.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def predict_x0_from_x_t_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_inv_alphas_bar, t, x_t.shape) * x_t - 
            extract(self.sqrt_inv_alphas_bar_m1, t, x_t.shape) * eps
        )
    
    def p_theta_mean_variance(self, x_t, t, clip_denoised: bool = True, return_x0: bool = False):
        assert t.shape[0] == x_t.shape[0] # batch size should be the same
        model_output = self.model(x_t, t)
        
        if self.var_type == "learned":
            raise NotImplementedError(self.var_type)
        elif self.var_type in ["fixedsmall", "fixedlarge"]:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas, self.log_betas),
                'fixedsmall': (self.posterior_variance, self.posterior_log_variance_clipped),
            }[self.var_type]
            # variance shape will be (bs, 3, size, size)
            model_variance = extract(model_variance, t, x_t.shape) * torch.ones(x_t.shape, device=x_t.device)
            model_log_variance = extract(model_log_variance, t, x_t.shape) * torch.ones(x_t.shape, device=x_t.device)
        else:
            raise NotImplementedError(self.var_type)
        
        # Mean parameterization
        _maybe_clip = lambda x_: (torch.clamp(x_, -1., 1.) if clip_denoised else x_)
        if self.mean_type == 'xprev':
            raise NotImplementedError(self.mean_type)
        elif self.mean_type == 'xstart':
            raise NotImplementedError(self.mean_type)
        elif self.mean_type == 'eps':  # the model predicts epsilon
            pred_xstart = _maybe_clip(self.predict_x0_from_x_t_eps(x_t = x_t, t = t, eps = model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(x_0 = pred_xstart, x_t = x_t, t = t)
        else:
            raise NotImplementedError(self.mean_type)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        
        if return_x0 == True:
            return model_mean, model_variance, model_log_variance, pred_xstart
        else:
            return model_mean, model_variance, model_log_variance

    def mse_loss(self, x_0, t, noise = None):
        assert t.shape == [x_0.shape[0]]
        
        if noise is None:
            noise = torch.randn_like(x_0)
        
        assert noise.shape == x_0.shape and noise.type == x_0.type
        
        x_t = self.q_sample(x_0, t, noise)
        return F.mse_loss(self.model(x_t, t), noise, reduction = 'none').mean((1, 2, 3))

    def KL_divergence(self, x_0, x_t, t, clip_denoised: bool = False, return_x0: bool = False):
        assert t.shape[0] == x_0.shape[0]
        assert x_0.shape == x_t.shape
        
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_0 = x_0, x_t = x_t, t = t)
        model_mean, _, model_log_variance, pred_x0 = self.p_theta_mean_variance(x_t = x_t, t = t, clip_denoised = clip_denoised, return_x0 = True)
        
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance).mean((1, 2, 3)) # / math.log(2.)
        
        decoder_nll = -discretized_gaussian_log_likelihood(
        x_0, means=model_mean, log_scales=0.5 * model_log_variance)
        assert decoder_nll.shape == x_0.shape
        
        decoder_nll = decoder_nll.mean((1, 2, 3)) # / math.log(2.)
        
        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where(t == 0, decoder_nll, kl)
        
        return (output, pred_x0) if return_x0 else output
    
    def calc_rate_distortion(self, x_0, device, clip_denoised: bool = True):
        KLs = []
        distortions = []
        with tqdm(range(self.T), desc="Rate & Distortion Computing") as progress:
            for time_step_ in progress:
                time_step = self.T - time_step_ - 1
                t = x_0.new_ones([x_0.shape[0], ], dtype=torch.long) * time_step
                
                kl, pred_x0 = self.KL_divergence(x_0, self.q_sample(x_0, t), t, clip_denoised = False, return_x0 = True)
                KLs.append(kl)
                distortions.append(torch.sqrt(F.mse_loss(x_0 * 0.5 + 0.5, torch.clamp(pred_x0 * 0.5 + 0.5, 0, 1), reduction='mean')))
        
        return torch.stack(KLs, dim = 0), torch.stack(distortions, dim = 0)
    
    # def sample_with_KL(self, batch_size, device):
    #     x_T = torch.randn(size=[batch_size, 3, 32, 32], device=device)
        
    #     x_t = x_T
    #     noises = []
    #     with tqdm(range(self.T), desc="Sampling Processing") as progress:
    #         for time_step_ in progress:
    #             time_step = self.T - time_step_ - 1
    #             t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
    #             mean, var, _ = self.p_theta_mean_variance(x_t=x_t, t=t, clip_denoised = False, return_x0 = False)
    #             # no noise when t == 0
    #             if time_step > 0:
    #                 noise = torch.randn_like(x_t)
    #             else:
    #                 noise = 0
    #             x_t = mean + torch.sqrt(var) * noise
                
    #             noises.append(noise)
                
    #             assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        
    #     x_0 = torch.clip(x_t, -1, 1)
    #     KLs = []
    #     distortions = []
        
    #     x_t = x_T
    #     with tqdm(range(self.T), desc="Rate & Distortion Computing") as progress:
    #         for time_step_ in progress:
    #             time_step = self.T - time_step_ - 1
    #             t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
    #             mean, var, _, pred_x0 = self.p_theta_mean_variance(x_t=x_t, t=t, clip_denoised = False, return_x0=True)
                
    #             KLs.append(self.KL_divergence(x_0, x_t, t))
    #             distortions.append(torch.sqrt(F.mse_loss(x_0 * 0.5 + 0.5, torch.clamp(pred_x0 * 0.5 + 0.5, 0, 1), reduction='mean')) * 255.)
                
    #             x_t = mean + torch.sqrt(var) * noises[time_step_]
                
    #     return x_0, torch.stack(KLs, dim = 0), torch.stack(distortions, dim = 0)

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='mean')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_bar))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_bar - 1))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def predict_x0_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - 
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t, return_x0 = False):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        if return_x0 == False:
            return xt_prev_mean, var
        else:
            return xt_prev_mean, var, self.predict_x0_mean_from_eps(x_t, t, eps)

    def forward(self, x_T, batch_id):
        """
        Algorithm 2.
        """
        x_t = x_T
        with tqdm(range(self.T), desc="Sampling Processing") as progress:
            progress.set_postfix(ordered_dict={
                "batch": batch_id
            })
            for time_step_ in progress:
                time_step = self.T - time_step_ - 1
                t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, var= self.p_mean_variance(x_t=x_t, t=t)
                # no noise when t == 0
                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                x_t = mean + torch.sqrt(var) * noise
                assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
    
    def progressive_sampling_and_save(self, x_T, paths: List[str], batch_id: int):
        x_t = x_T
        for index, path in enumerate(paths):
            Path(path).mkdir(parents=True, exist_ok=True)
            save_image(torch.clamp(x_t[index] * 0.5 + 0.5, 0, 1), os.path.join(path, "00.png"))
        with tqdm(range(self.T), desc="Sampling Processing") as progress:
            progress.set_postfix(ordered_dict={
                "batch": batch_id
            })
            for time_step_ in progress:
                time_step = self.T - time_step_ - 1
                t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, var, pred_x0 = self.p_mean_variance(x_t=x_t, t=t, return_x0 = True)
                # no noise when t == 0
                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                x_t = mean + torch.sqrt(var) * noise
                assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
                
                if time_step % 50 == 0:
                    for index, path in enumerate(paths):
                        save_image(torch.clamp(pred_x0[index] * 0.5 + 0.5, 0, 1), os.path.join(
                            path,  str(20 - time_step // 50).zfill(2) + ".png"))
                

        x_0 = x_t
        return torch.clamp(x_0, -1, 1)


