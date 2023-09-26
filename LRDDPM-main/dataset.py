import os
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random


class Data:
    # 数据加载类
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True):

        train_path = os.path.join(self.config.data.train_data_dir)
        val_path = os.path.join(self.config.data.test_data_dir)

        train_dataset = MyDataset(train_path,
                                  n=self.config.training.patch_n,
                                  patch_size=self.config.data.image_size,
                                  transforms=self.transforms,
                                  parse_patches=parse_patches)
        val_dataset = MyDataset(val_path,
                                n=self.config.training.patch_n,
                                patch_size=self.config.data.image_size,
                                transforms=self.transforms,
                                parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        # 训练数据
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        # 评估数据
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


# 数据集加载类
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, parse_patches=True):
        super().__init__()

        # 数据集目录
        self.dir = dir

        # 获取输入图像文件名列表和目标文件名列表
        input_names = os.listdir(dir+'input')
        gt_names = os.listdir(dir+'target')

        # 存储文件名列表和其他参数
        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    # 静态方法：获取随机裁剪参数
    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    # 静态方法：对图像多次随机裁剪
    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    # 方法：获取图像对
    def get_images(self, index):
        # 获取输入图像和目标图像
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        # 获取图像id
        img_id = re.split('/', input_name)[-1][:-4]
        # 读取图像
        input_img = PIL.Image.open(os.path.join(self.dir, 'input', input_name)) if self.dir else PIL.Image.open(input_name)
        try:
            gt_img = PIL.Image.open(os.path.join(self.dir, 'target', gt_name)) if self.dir else PIL.Image.open(gt_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.dir, 'target', gt_name)).convert('RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')

        if self.parse_patches:
            # 如果需要解析裁剪图像
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            # 对图像进行多次随机裁剪
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            # 对每对裁剪后的图像应用转换并拼接为一个张量
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
                       for i in range(self.n)]
            # 返回裁剪后的图像张量和图像id
            return torch.stack(outputs, dim=0), img_id
        else:
            # 如果不需要解析裁剪图像，将图像调整为 16 的倍数以进行整体图像修复
            wd_new, ht_new = input_img.size
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))
            # 调整图像大小
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)

            # 对调整大小后的图像应用转换并拼接为一个张量
            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
