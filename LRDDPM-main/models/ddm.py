import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import utils
from models.unet import DiffusionUNet
import tqdm


def data_transform(X):
    # 将数据映射到[-1, 1]之间
    return 2 * X - 1.0


def inverse_data_transform(X):
    # 将数据逆向映射回到[0, 1]之间
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    '''
    指数移动平均辅助类
    * 用于模型参数的指数移动平均
    * 支持参数注册、更新、EMA（替换模型参数为移动平均值）以及复制替代模型

    没太看懂，过会儿再看
    '''

    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        """
        如果module是nn.DataParallel类型的对象，需要通过module.module获取实际的模型对象，因为nn.DataParallel对模型进行了包装。
        遍历模型module的所有参数，使用named_parameters()方法获取参数的名称和值。
        对于具有requires_grad=True的参数，进行参数更新的计算。
        使用指数移动平均方法更新参数的阴影变量（shadow variable），阴影变量用于保存参数的移动平均值。
        更新参数的计算公式为：self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data.to('cuda')。其中，self.mu是移动平均的衰减因子，param.data是当前参数的值，self.shadow[name].data是参数的阴影变量的值。
        更新后的阴影变量将被用于计算模型预测和损失函数。
        通过使用指数移动平均方法，模型参数的更新可以平滑参数的变化，有助于提高模型的泛化能力和稳定性。这在训练过程中对模型进行参数更新时是一种常用的技术。
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data.to('cuda')

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    '''
    获取beta值
    :param beta_schedule: beta值计算方式
    :param beta_start: beta初始值
    :param beta_end: beta结束值
    :param num_diffusion_timesteps: 扩散步数
    :return: beta值
    '''

    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas



def maximum_contrast_enhancement(image):
    enhanced_image = np.zeros_like(image, dtype=np.float32)
    padded_image = cv2.copyMakeBorder(image, 7, 7, 7, 7, cv2.BORDER_REFLECT)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch = padded_image[i:i + 3, j:j + 3]
            local_contrast_values = []
            for m in range(i, i + 7):
                for n in range(j, j + 7):
                    local_patch = padded_image[m:m + 3, n:n + 3]
                    local_contrast_values.append(np.square(local_patch - patch).mean())
            max_contrast = max(local_contrast_values)
            enhanced_image[i, j] = max_contrast

    enhanced_image = (enhanced_image - np.min(enhanced_image)) / (np.max(enhanced_image) - np.min(enhanced_image))
    enhanced_image = (enhanced_image * 255).astype(np.uint8)

    return enhanced_image


def noise_estimation_loss(model, x0, t, e, b):
    # 计算暗通道
    dark_ch = dark_channel(x0[:, :3, :, :], window_size=15)
    output = dark_ch
    print("output", output.shape)
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)  # 通过扩散步数 t 计算权重信息
    x = x0[:, :3, :, :] * a.sqrt() + e * (1.0 - a).sqrt()  # 计算噪声估计值
    x0_np = x0[:, :3, :, :].cpu().numpy()
    print("x0_np.shape", x0_np.shape)
    enhanced_images_tensor = []
    for i in range(x0_np.shape[0]):
        enhanced_image = maximum_contrast_enhancement(x0_np[i].transpose(1, 2, 0))
        enhanced_image_tensor = torch.from_numpy(enhanced_image).unsqueeze(0).permute(0, 3, 1, 2).float()
        enhanced_images_tensor.append(enhanced_image_tensor)

    enhanced_images_tensor = torch.cat(enhanced_images_tensor, dim=0)
    enhanced_images_tensor = enhanced_images_tensor.to(x0.device)
    print(enhanced_images_tensor.shape)
    plt.subplot(1, 3, 1)
    input_img = x0[0, :3].cpu().numpy().transpose(1, 2, 0)  # 重新排列维度
    input_img = torch.from_numpy(input_img)  # 将numpy.ndarray转换为Tensor
    input_img = torch.clamp(input_img, 0, 1)  # 将图像数据限制在[0, 1]范围内
    plt.imshow(input_img)
    plt.title('Input Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    output_img = output[0].cpu().numpy()
    output_img = output_img.transpose(1, 2, 0)
    output_img = torch.from_numpy(output_img)
    output_img = torch.clamp(output_img, 0, 1)
    plt.imshow(output_img, cmap='gray')
    plt.title('Dark Channel')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    enhanced_img = enhanced_images_tensor[0].squeeze().cpu().numpy().transpose(1, 2, 0)
    enhanced_img = torch.from_numpy(enhanced_img)  # 将numpy.ndarray转换为Tensor
    enhanced_img = torch.clamp(enhanced_img, 0, 255).byte()  # 将图像数据限制在[0, 255]范围内，并转换为字节类型（uint8）
    plt.imshow(enhanced_img, cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')
    plt.show()

    input_model = torch.cat([x0[:, :3, :, :], x, output, enhanced_images_tensor], dim=1)
    print("input_model", input_model.shape)
    output = model(input_model, t.float())  # 计算输出图像
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def dark_channel(image, window_size):
    print("Image shape: ", image.shape)
    min_pool = nn.MaxPool2d(window_size, stride=1, padding=window_size // 2)
    dark_channel = min_pool(-image)
    # output = -dark_channel
    output = -torch.min(dark_channel, dim=1)[0].unsqueeze(1)
    return output


class DenoisingDiffusion(object):
    '''
    去噪扩散模型
    '''

    def __init__(self, config):
        # 初始化模型训练配置
        super().__init__()
        self.config = config
        self.device = config.device

        # 创建 DiffusionUNet 模型并配置相关参数
        self.model = DiffusionUNet(config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        # 创建 EMAHelper 辅助类并注册模型
        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        # 创建优化器、学习率调度器
        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        # 获取扩散时间步数的 beta 值
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        # 将 beta 值转换为 tensor 并保存
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=False):
        # 加载预模型权重
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, DATASET):
        # 执行模型训练过程，包括数据加载、模型更新、模型保存等
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        # 如果有断点，则加载断点
        if os.path.isfile(self.config.training.resume):
            self.load_ddm_ckpt(self.config.training.resume)

        # 开始训练
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('=> current epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                # 数据预处理
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()  # 设置模型为训练模式
                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)
                e = torch.randn_like(x[:, 3:, :, :])
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = noise_estimation_loss(self.model, x, t, e, b)

                if self.step % 10 == 0:
                    print(
                        'step: %d, loss: %.6f, time consumption: %.6f' % (self.step, loss.item(), data_time / (i + 1)))

                # 更新参数
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                # 保存模型
                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'config': self.config
                    }, filename=self.config.training.resume)

    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        '''
        采样图像：根据条件图像和输入图像采样生成图像，用于生成验证图像
        :param x_cond: 条件图像
        :param x: 输入图像
        :param last: 是否只返回最后一张图像
        :param patch_locs: 图像块位置
        :param patch_size: 图像块大小
        :return: 采样图像
        '''
        skip = self.config.diffusion.num_diffusion_timesteps // self.config.sampling.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
        else:
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs

    def sample_validation_patches(self, val_loader, step):
        '''
        采样验证图像：根据验证数据集采样生成图像，保存条件图像和生成的去噪图像
        :param val_loader: 验证数据加载器
        :param step: 当前步数
        '''
        image_folder = os.path.join(self.config.data.val_save_dir, str(self.config.data.image_size))
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = x[:, :3, :, :].to(self.device)  # 条件图像
            x_cond = data_transform(x_cond)
            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond, x)
            x = inverse_data_transform(x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                utils.logging.save_image(x_cond[i], os.path.join(image_folder, str(step), f"{i}_cond.png"))
                utils.logging.save_image(x[i], os.path.join(image_folder, str(step), f"{i}.png"))
