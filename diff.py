import torch
import numpy as np 

def gradient(y, x, grad_outputs=None):
    " Compute dy/dx @ grad_outputs "
    " train_points: [B, DIM]"
    " model: R^{DIM} --> R"
    " grads: [B, DIM]"
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grads = torch.autograd.grad(y, [x], 
                                grad_outputs=grad_outputs,                        
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)[0]
    
    # torch.nn.utils.clip_grad_norm_(x, max_norm=2.0, norm_type=2)
    return grads

def partial_t_j(f, x, t, j):
    """
    :param s: function R^N -> R^N
    :param x: torch.tensor of shape [B, N]
    :return: (dsdt)_j (torch.tensor) of shape [B, 1]
    """
    assert j <= x.shape[-1]
    s = f(x, t)
    v = torch.zeros_like(s)
    v[:, j] = 1.
    dy_j_dx = torch.autograd.grad(
                   s,
                   t,
                   grad_outputs=v,
                   retain_graph=True,
                   create_graph=True,
                   allow_unused=False)[0]  # shape [B, N]
    
    torch.nn.utils.clip_grad_norm_(t, max_norm=2.0, norm_type=2)
    return dy_j_dx

def batch_div(f, x, t):
    x.requires_grad = True
    def batch_jacobian():
        f_sum = lambda x: torch.sum(f(x, t), axis=0)
        return torch.autograd.functional.jacobian(f_sum, x, create_graph=True, strict=True).permute(1,0,2) 
    jac = batch_jacobian()
    return torch.sum(jac.diagonal(offset=0, dim1=-1, dim2=-2), dim=-1, keepdim=False)
       

    
def hutch_div(score_model, sample, time_steps):      
    """Compute the divergence of the score-based model with Skilling-Hutchinson."""
    with torch.enable_grad():
        sample.requires_grad_(True)
        repeat = 1
        divs = torch.zeros((sample.shape[0],), device=sample.device, requires_grad=False) #div: [B,]
        for _ in range(repeat):
            epsilon = torch.randn_like(sample)
            score_e = torch.sum(score_model(sample, time_steps) * epsilon)
            grad_score_e = torch.autograd.grad(score_e, sample,
                                                retain_graph=True,
                                                create_graph=True,
                                                allow_unused=False)[0]
            divs += torch.sum(grad_score_e * epsilon, dim=(1))  
        divs = divs/repeat
    return divs


# def t_finite_diff(fn, x, t, hs=0.001, hd=0.0005):
#     up = hs**2 * fn(x, torch.clamp((t + hd), max=1)) + (hd**2 - hs**2) * fn(x, t) - hd**2 * fn(x, torch.clamp((t - hs), min=1e-5))
#     print("up", up)
#     low = hs * hd * (hd+hs)
#     print("low", low)
#     return up/low  

def t_finite_diff(fn, x, t, hs=0.001, hd=0.0005):
    up1 = (hs/hd) * fn(x, torch.clamp((t + hd), max=1))
    low1 =  (hd+hs)
    up2 = ((hd - hs) / hs) * fn(x, t)
    low2 = hd 
    up3 = (hd/hs) * fn(x, torch.clamp((t - hs), min=1e-5))
    low3 = (hd+hs)
    item1 = (up1/low1)
    item2 = (up2/low2)
    item3 = (up3/low3)
    if torch.isnan(item1).any() != False:
        torch.save(item1, "item1.pt")
        print("item1", item1)
    if torch.isnan(item2).any() != False:
        torch.save(item2, "item2.pt")
        print("item2", item2)
        
    if torch.isnan(item3).any() != False:
        torch.save(item3, "item3.pt")
        print("item3", item3)
    
    return item1  +  item2  -  item3
