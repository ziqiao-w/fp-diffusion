# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE
import logging


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5,
                    scalar_fp='both', fp_dist='pert', fp_wgt_type='ll', 
                    alpha=0.0, beta=0.001, m=2):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
  # print("sde loss")
  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    # print("alpha", alpha)
    # print("beta", beta)
    if np.isclose(alpha, 0.0, atol=1e-04):
        loss = loss
        # print("no fpe")
    elif not np.isclose(alpha, 0.0, atol=1e-04):  
        # print("fpe used")
        if fp_dist == 'random':   
            x_low = batch.min()
            x_high = batch.max() 
            xx = torch.rand_like(batch).reshape((batch.shape[0], -1)) 
            xx = (x_high - x_low) * xx + x_low
        elif fp_dist == 'pert':
            xx = perturbed_data.reshape((batch.shape[0], -1)) 
        elif fp_dist == 'x':
            xx = batch.reshape((batch.shape[0], -1))  
        
        if scalar_fp in ("True"):
            assert not np.isclose(beta, 0.0, atol=1e-04)
            FP = sde.PDE(score_fn, scalar_fp='True', wgt=fp_wgt_type, m=m).fp(xx, t) 
            print("fpe: {}".format(FP.detach().cpu().numpy()))
            loss = loss + beta * FP
            
        elif scalar_fp in ("False"):
            assert not np.isclose(alpha, 0.0, atol=1e-04) 
            vec_fpe = sde.PDE(score_fn, scalar_fp='False', wgt=fp_wgt_type, m=m).fp(xx, t)
            # print("vec_fpe", vec_fpe)
            print("vec_fpe", vec_fpe)
            if torch.isnan(vec_fpe).any() != False:
              torch.save(t, 't_nan.pt')
              torch.save(xx, 'xx_nan.pt')
              torch.save(vec_fpe, 'vec_fpe_nan.pt')
            assert torch.isnan(vec_fpe).any() == False, "Loss.py Tensor contains NaN values"
            loss = loss + alpha * vec_fpe
        
        elif scalar_fp in ("both"):
            assert not np.isclose(alpha, 0.0, atol=1e-04) and not np.isclose(beta, 0.0, atol=1e-04)
            scalar_fpe, vec_fpe = sde.PDE(score_fn, scalar_fp='both', wgt=fp_wgt_type, m=m).fp(xx, t)
            print("vec_fpe", vec_fpe)
            if torch.isnan(vec_fpe).any() != False:
              torch.save(t, 't_nan.pt')
              torch.save(xx, 'xx_nan.pt')
              torch.save(vec_fpe, 'vec_fpe_nan.pt')
            assert torch.isnan(vec_fpe).any() == False, "Loss.py Tensor contains NaN values"
            # print("scalar_fpe", scalar_fpe)
            # logging.info("scalar_fpe: {}".format(scalar_fpe.detach().cpu().numpy()))
            # logging.info("vec_fpe: {}".format(vec_fpe.detach().cpu().numpy()))
            # loss = loss + (alpha * vec_fpe + beta * scalar_fpe)
            return loss, (alpha * vec_fpe + 0 * scalar_fpe)
    return loss

  return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
  print("smld loss chosen")
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]
    perturbed_data = noise + batch
    score = model_fn(perturbed_data, labels)
    target = -noise / (sigmas ** 2)[:, None, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."
  print("ddpm loss chosen")
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False,
                scalar_fp='both', fp_dist='pert', fp_wgt_type='ll', 
                alpha=0.0, beta=0.001, m=2):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting,
                              scalar_fp=scalar_fp, fp_dist=fp_dist, fp_wgt_type=fp_wgt_type, 
                              alpha=alpha, beta=beta, m=m)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      if scalar_fp=='both':
        loss, fpe = loss_fn(model, batch)
        fpe.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        # print(list(model.parameters())[-1].grad)
        fpe_grad = []
        for param in model.parameters():
          fpe_grad.append(param.grad.clone())
        optimizer.zero_grad()
        # print(list(model.parameters())[-1].grad)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        # print(list(model.parameters())[-1].grad)
        idx = 0
        for param in model.parameters():
          param.grad += 1 * fpe_grad[idx]
          idx += 1
        # print(list(model.parameters())[-1].grad)
      else:
        loss = loss_fn(model, batch)
        loss.backward()
      # for param in model.parameters():
      #   if param.grad is not None:
      #     print(param.grad)
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())
      # ema = state['ema']
      # ema.store(model.parameters())
      # ema.copy_to(model.parameters())
      # loss = loss_fn(model, batch)
      # ema.restore(model.parameters())
    return loss

  return step_fn
