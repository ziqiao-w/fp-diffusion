import torch
from torch import nn, einsum
import torch.nn.functional as F
from PIL import Image
import requests
from tqdm import tqdm
from torch.autograd import Variable

import diff
from dataset import channels, dataloader, image_size
from unet import Unet
from utils import extract, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, betas, sqrt_recip_alphas, \
    posterior_variance, timesteps

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)  # PIL image of shape HWC
# image.show()

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="logger")


transform = Compose([
    Resize(image_size),
    CenterCrop(image_size),
    ToTensor(),  # turn into torch Tensor of shape CHW, divide by 255
    Lambda(lambda t: (t * 2) - 1),

])

# x_start = transform(image).unsqueeze(0)
# print(x_start.shape)

import numpy as np

reverse_transform = Compose([
    Lambda(lambda t: (t + 1) / 2),
    Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
    Lambda(lambda t: t * 255.),
    Lambda(lambda t: t.numpy().astype(np.uint8)),
    ToPILImage(),
])

# x_start = reverse_transform(x_start.squeeze())


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def get_noisy_image(x_start, t):
    # add noise
    x_noisy = q_sample(x_start, t=t)

    # turn back into PIL image
    noisy_image = reverse_transform(x_noisy.squeeze())

    return noisy_image


# take time step
# t = torch.tensor([40])

# get_noisy_image(x_start, t).show()

import matplotlib.pyplot as plt

# use seed for reproducability
# torch.manual_seed(0)


# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
# def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
#     if not isinstance(imgs[0], list):
#         # Make a 2d grid even if there's just 1 row
#         imgs = [imgs]
#
#     num_rows = len(imgs)
#     num_cols = len(imgs[0]) + with_orig
#     fig, axs = plt.subplots(figsize=(200, 200), nrows=num_rows, ncols=num_cols, squeeze=False)
#     for row_idx, row in enumerate(imgs):
#         row = [image] + row if with_orig else row
#         for col_idx, img in enumerate(row):
#             ax = axs[row_idx, col_idx]
#             ax.imshow(np.asarray(img), **imshow_kwargs)
#             ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#
#     if with_orig:
#         axs[0, 0].set(title='Original image')
#         axs[0, 0].title.set_size(8)
#     if row_title is not None:
#         for row_idx in range(num_rows):
#             axs[row_idx, 0].set(ylabel=row_title[row_idx])
#
#     plt.tight_layout()
#     plt.show()


# plot([get_noisy_image(x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 199]])

step = 0
def p_losses(denoise_model, x_start, t1, noise=None, loss_type="l1"):
    t = t1.clone().to(torch.float32)
    t.requires_grad = True
    if noise is None:
        noise = torch.randn_like(x_start)
    x_start.requires_grad = True
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    # def func(t2):
    #     return denoise_model(x_noisy, t2) / (-torch.sqrt(betas_t))

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    elif loss_type == "FPE":

        def flat_model(x, t_):
            x = x.reshape((x.shape[0], channels, image_size, image_size))
            std = extract(sqrt_one_minus_alphas_cumprod, t_, x.shape).reshape((x.shape[0], -1))
            true_noise = denoise_model(x, t_).reshape((x.shape[0], -1))
            return - (true_noise / std)

        def sde(x, t_):
            beta_0 = 0.1
            beta_1 = 20
            beta_t = beta_0 + t_ * (beta_1 - beta_0)
            drift = -0.5 * beta_t[:, None, None, None] * x
            diffusion = torch.sqrt(beta_t)
            return drift, diffusion
        # score = (predicted_noise / (-torch.sqrt(betas_t))).requires_grad_(True)
        # compute RHS
        x_noisy = x_noisy.reshape((x_noisy.shape[0], -1))
        score = flat_model(x_noisy, t)
        t.requires_grad = True
        # first calculate s_l22
        s_l22 = torch.linalg.norm(score, 2, dim=1, keepdim=False) ** 2
        # (Batchsize, )

        # calculate div_score
        div_score = diff.hutch_div(flat_model, x_noisy, t)
        # (Batchsize, )

        _, diffusion = sde(x_noisy, t / (timesteps - 1))
        g_pow = (diffusion[:, None])**2
        f_dot_s = torch.einsum('bs,bs->b', x_noisy, score)

        # gradient for x
        RHS = (g_pow / 2) * diff.gradient(div_score + s_l22 + f_dot_s, x_noisy)

        #compute LHS
        dsdt = diff.t_finite_diff(flat_model, x_noisy, t)
        error = (dsdt - RHS)
        res = torch.linalg.norm(error, ord=2, dim=1)
#         print("dsdt", torch.linalg.norm(dsdt, ord=2, dim=1).mean())
#         print("RHS", torch.linalg.norm(RHS, ord=2, dim=1).mean())
        loss1 = torch.mean(res) / x_noisy.shape[-1]
#         x_noisy1 = q_sample(x_start=x_start, t=torch.ones_like(t), noise=noise)
#         loss2 = torch.mean(torch.linalg.norm(noise - denoise_model(x_noisy1, torch.ones_like(t)), ord=2, dim=1))
#         loss2 = F.mse_loss(noise, denoise_model(x_noisy1, torch.ones_like(t)))
#         std = extract(sqrt_one_minus_alphas_cumprod, t, x_noisy.shape).reshape((x_noisy.shape[0], -1))
#         loss2 = torch.mean(torch.norm(score + noise.reshape((noise.shape[0], -1))/std, 2, dim=1))
        loss2 = F.mse_loss(noise, predicted_noise)
        loss = 0.15*loss1 + 1 * loss2
#         print("FP:", loss1.item())
        writer.add_scalar('FP', loss1.item(), step)
#         print("ScoreMatching:", loss2.item())
        writer.add_scalar('DSM', loss2.item(), step)
        # term4 = torch.zeros_like(x_start)
        # for i in range(score.shape[0]):
        #     for j in range(score.shape[1]):
        #         for k in range(score.shape[2]):
        #             for l in range(score.shape[3]):
        #                 value = torch.autograd.grad(outputs=score[i, j, k, l], inputs=t[i:i + 1], retain_graph=True,
        #                                             allow_unused=True)[0]
        #                 term4[i, j, k, l] = value[0]
        #                 print(i, j, k, l)
        #
        # dot = torch.einsum('bijk,bijk->', x_start.requires_grad_(True), score.requires_grad_(True))
        # dot.backward(retain_graph=True)
        # term1 = x_start.grad.clone()
        # x_start.grad = None
        # score.grad = None
        #
        #
        #
        # g = torch.zeros_like(x_start)
        # for k in range(0, Nest):
        #     v = torch.randn_like(x_start)
        #     dot1 = torch.einsum('bijk,bijk->', v.requires_grad_(True), score.requires_grad_(True))
        #     dot1.backward(retain_graph=True)
        #     dot2 = torch.einsum('bijk,bijk->', v.requires_grad_(True), x_start.grad.requires_grad_(True))
        #     dot2.backward(retain_graph=True)
        #     dot2_grad = x_start.grad.clone()
        #     x_start.grad = None
        #     g = g + dot2_grad
        # term2 = g / Nest
        # loss1 = 4 * F.mse_loss(0.5 * betas_t * (term1 + term2 + term3), term4)
        #
        #
        #
    else:
        raise NotImplementedError()

    return loss


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)


@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


from pathlib import Path


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


results_folder = Path("./results")
results_folder.mkdir(exist_ok=True)
save_and_sample_every = 300

from torch.optim import Adam

device = "cuda:1" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,),
    # dim_mults=(1, 2,),
)
model.to(device)
model.load_state_dict(torch.load("results/model-0-5.pth"))
optimizer = Adam(model.parameters(), lr=0.0005)

from torchvision.utils import save_image

epochs = 6
for epoch in range(epochs):
    for step_, batch in enumerate(dataloader):
        optimizer.zero_grad()
        step += 1
        batch_size = batch["pixel_values"].shape[0]
        batch = batch["pixel_values"].to(device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = p_losses(model, batch, t, loss_type="FPE")
        if step % 100 == 0:
            print("Loss:", loss.item())
            writer.add_scalar('Loss', loss.item(), step)
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            true_noise_loss = p_losses(model, batch, t, loss_type="l2")
            print("true denoise loss: ", true_noise_loss.item())
            writer.add_scalar('noise_l2', true_noise_loss.item(), step)

        # save generated images
        if step != 0 and step % save_and_sample_every == 0:
            milestone = step // save_and_sample_every
            batches = num_to_groups(4, batch_size)
            print("sample")
            all_images_list = list(
                map(lambda n: sample(model, batch_size=n, channels=channels, image_size=image_size), batches))
            for i in range(len(all_images_list)):
                all_images_list[i] = torch.Tensor(all_images_list[i])
            print(all_images_list[0].shape)
            all_images = torch.cat(all_images_list, dim=0)
            print(all_images.shape)
            all_images = (all_images + 1) * 0.5

            save_image(all_images[-1], str(results_folder / f'sample-{epoch}-{milestone}.png'), nrow=4)
            torch.save(model.state_dict(), str(results_folder / f'model-{epoch}-{milestone}.pth'))
writer.close()