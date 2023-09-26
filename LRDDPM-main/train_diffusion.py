# -*- coding: utf-8 -*-
import argparse
import os
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from dataset import Data
from models import DenoisingDiffusion


def config_get():
    # 参数配置：获取配置信息并返回一个包含配置信息的对象
    parser = argparse.ArgumentParser()
    # 参数配置文件路径
    parser.add_argument("--config", default='configs.yml', type=str, required=False, help="Path to the config file")
    args = parser.parse_args()

    with open(os.path.join(args.config), "r",encoding="utf-8") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return new_config


def dict2namespace(config):
    # 将配置信息转换为命名空间
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():

    # 获取配置信息
    config = config_get()

    # 判断是否使用 cuda
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("=> using device: {}".format(device))
    config.device = device

    # 随机种子
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = True

    # 加载数据
    DATASET = Data(config)
    _, val_loader = DATASET.get_loaders()

    # 创建模型
    print("=> creating denoising diffusion model")
    diffusion = DenoisingDiffusion(config)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
