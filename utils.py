import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
import diff




def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


timesteps = 1000

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def get_score_fn(sde, model, train=False, continuous=False):
  model_fn = get_model_fn(model, train=train)

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        score = model_fn(x, labels)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score = model_fn(x, labels)
        std = sqrt_one_minus_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None, None, None]
      return score

  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model_fn(x, labels)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu().to(torch.long))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class VPSDE():
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.name = 'vpsde'
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return 1



    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G

    def PDE(self, score_fn,
            wgt='constant',
            normalize=True,
            reduction='mean',
            list_dim=[],
            time_est='approx',
            div_est='approx',
            train=True,
            scalar_fp="False",
            m=1):

        sde_fn = self.sde
        flat_model = Merge(score_fn)

        class score_FP(self.__class__):
            def __init__(self):
                self.flat_model = flat_model
                self.normalize = normalize
                self.reduction = reduction
                self.list_dim = list_dim
                self.time_est = time_est
                self.div_est = div_est
                self.train = train
                self.wgt = wgt
                self.scalar_fp = scalar_fp
                self.m = m

            def fp(self, x, t):
                x = x.reshape((x.shape[0], -1))
                x.requires_grad = True
                t.requires_grad = True
                D = x.shape[-1]
                if self.list_dim == []:
                    self.list_dim = range(D)
                s = self.flat_model(x, t)
                if self.div_est == 'exact':
                    div_s = diff.batch_div(self.flat_model, x, t)
                elif self.div_est == 'approx':
                    div_s = diff.hutch_div(self.flat_model, x, t)
                s_l22 = torch.linalg.norm(s, 2, dim=1, keepdim=False) ** 2

                "Computing RHS"
                _, diffusion = sde_fn(x, t)

                if self.scalar_fp in ("True"):
                    g_pow = diffusion ** 2
                    f_dot_s = torch.einsum('bs,bs->b', x, s)
                    RHS = (g_pow / 2) * (div_s + s_l22 + f_dot_s)
                    RHS = RHS if self.train else RHS.cpu().detach().numpy()
                    res = torch.linalg.norm(RHS, ord=2) if self.train \
                        else np.linalg.norm(RHS, ord=2)
                elif self.scalar_fp in ("False"):
                    print("scalar_fp", self.scalar_fp)
                    g_pow = (diffusion[:, None]) ** 2
                    f_dot_s = torch.einsum('bs,bs->b', x, s)
                    RHS = (g_pow / 2) * diff.gradient(div_s + s_l22 + f_dot_s, x)
                    RHS = RHS if self.train else RHS.cpu().detach().numpy()

                    "Computing LHS"
                    if self.time_est == 'exact':
                        res = torch.zeros_like(t) if self.train else np.zeros_like(t.cpu().detach().numpy())
                        for j in list_dim:
                            dsdt = diff.partial_t_j(self.flat_model, x, t, j)
                            dsdt = dsdt if self.train else dsdt.cpu().detach().numpy()
                            residue = torch.clip((dsdt - RHS[0:, j]) ** 2, max=1.0e+30) if self.train \
                                else np.clip((dsdt - RHS[0:, j]) ** 2, a_min=0.0, a_max=1.0e+30)
                            res += residue  # batch square-sum (of each coordinate); shape: [B,]
                        res = torch.sqrt(res) if self.train else np.sqrt(res)
                    elif self.time_est == 'approx':
                        dsdt = diff.t_finite_diff(self.flat_model, x, t)
                        dsdt = dsdt if self.train else dsdt.cpu().detach().numpy()
                        error = (dsdt - RHS)
                        res = torch.linalg.norm(error, ord=2, dim=1) if self.train \
                            else np.linalg.norm((dsdt - RHS), ord=2, axis=1)
                        print("dsdt", torch.linalg.norm(dsdt, ord=2, dim=1).mean())
                        print("RHS", torch.linalg.norm(RHS, ord=2, dim=1).mean())

                        res = res ** self.m
                    if self.wgt == 'convention':
                        res = diffusion * res
                    elif self.wgt == 'll':
                        g_2 = diffusion ** 2
                        res = g_2 * res
                    elif self.wgt == 'constant':
                        res = res
                    else:
                        print("Undefined time weighting...")

                    if self.reduction == 'mean':
                        res = torch.mean(res) if self.train else np.mean(res)
                    elif self.reduction == 'sum':
                        res = res.sum() if self.train else np.sum(res)
                    elif self.reduction == 'batchwise':
                        res = res
                    elif self.reduction == 'pointwise':
                        res = error
                    else:
                        print("Undefined reduction method...")
                    if self.normalize:
                        res = res / (D ** self.m)

                elif self.scalar_fp in ("both"):
                    g_pow = diffusion ** 2
                    f_dot_s = torch.einsum('bs,bs->b', x, s)
                    RHS = (g_pow / 2) * (div_s + s_l22 + f_dot_s)
                    RHS = RHS if self.train else RHS.cpu().detach().numpy()
                    scalar_res = torch.linalg.norm(RHS, ord=2) if self.train else np.linalg.norm(RHS, ord=2)

                    print("scalar_fp", self.scalar_fp)
                    g_pow = (diffusion[:, None]) ** 2
                    f_dot_s = torch.einsum('bs,bs->b', x, s)
                    D = x.shape[-1]
                    if self.list_dim == []:
                        self.list_dim = range(D)
                    RHS = (g_pow / 2) * diff.gradient(div_s + s_l22 + f_dot_s, x)

                    "Computing LHS"
                    dsdt = diff.t_finite_diff(self.flat_model, x, t)
                    error = (dsdt - RHS)
                    res = torch.linalg.norm(error, ord=2, dim=1)
                    res = res ** self.m
                    if self.wgt == 'convention':
                        res = diffusion * res
                    elif self.wgt == 'll':
                        g_2 = diffusion ** 2
                        res = g_2 * res
                    elif self.wgt == 'constant':
                        res = res
                    else:
                        print("Undefined time weighting...")

                    if self.reduction == 'mean':
                        res = torch.mean(res)
                    elif self.reduction == 'sum':
                        res = res.sum()
                    elif self.reduction == 'batchwise':
                        res = res
                    elif self.reduction == 'pointwise':
                        res = error
                    else:
                        print("Undefined reduction method...")

                    if self.normalize:
                        res = res / (D ** self.m)
                    vec_res = res

                x.requires_grad = False
                t.requires_grad = False

                if self.scalar_fp in ("True", "False"):
                    print("FP type {}: {}".format(self.scalar_fp, res.detach().cpu().numpy()), flush=True)
                    return res
                elif self.scalar_fp in ("both"):
                    return scalar_res, vec_res

        return score_FP()