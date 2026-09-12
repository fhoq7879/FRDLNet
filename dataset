# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os

identity = lambda x: x


class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

        # 添加：验证所有文件路径是否存在
        self._validate_all_paths()

    def _validate_all_paths(self):
        """验证所有图像路径是否存在"""
        missing_files = []
        for i, image_path in enumerate(self.meta['image_names']):
            if not os.path.exists(image_path):
                missing_files.append((i, image_path))

        if missing_files:
            print(f"警告: 发现 {len(missing_files)} 个缺失文件")
            for i, path in missing_files[:5]:  # 只显示前5个缺失文件
                print(f"  索引 {i}: {path}")

    def __getitem__(self, i):
        original_image_path = self.meta['image_names'][i]

        # 转换为绝对路径并验证
        image_path = self._resolve_image_path(original_image_path)

        if image_path is None:
            # 文件不存在，返回默认图像
            print(f"警告: 文件不存在，使用默认图像: {original_image_path}")
            img = self._get_default_image()
        else:
            try:
                img = Image.open(image_path).convert('RGB')
                img = self.transform(img)
            except Exception as e:
                print(f"无法加载图像 {image_path}: {e}")
                img = self._get_default_image()

        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def _resolve_image_path(self, image_path):
        """解析图像路径，确保使用绝对路径"""
        # 如果已经是绝对路径且存在，直接返回
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path

        # 尝试作为相对路径处理
        base_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            image_path,  # 原始路径
            os.path.join(base_dir, image_path),  # 相对于当前脚本的路径
            os.path.join(base_dir, '..', 'filelists', image_path)  # 相对于filelists目录
        ]

        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path

        return None

    def _get_default_image(self):
        """返回默认图像（黑色图像）"""
        return torch.zeros(3, 84, 84)  # 根据您的图像尺寸调整

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.cl_list = np.unique(self.meta['image_labels']).tolist()
        self.sub_meta = {}

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        # 关键修改：设置num_workers=0避免多进程路径问题
        sub_data_loader_params = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # 禁用多进程，避免工作目录问题
            pin_memory=False
        )

        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    # 添加缺失的 __len__ 方法
    def __len__(self):
        """返回数据集中类的数量（episode的数量）"""
        return len(self.cl_list)

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))


class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        original_image_path = self.sub_meta[i]

        # 使用路径解析方法
        image_path = self._resolve_image_path(original_image_path)

        if image_path is None:
            print(f"警告: 文件不存在，使用默认图像: {original_image_path}")
            img = self._get_default_image()
        else:
            try:
                img = Image.open(image_path).convert('RGB')
                img = self.transform(img)
            except Exception as e:
                print(f"无法加载图像 {image_path}: {e}")
                img = self._get_default_image()

        target = self.target_transform(self.cl)
        return img, target

    def _resolve_image_path(self, image_path):
        """解析图像路径（与SimpleDataset相同的方法）"""
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path

        base_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            image_path,
            os.path.join(base_dir, image_path),
            os.path.join(base_dir, '..', 'filelists', image_path)
        ]

        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path

        return None

    def _get_default_image(self):
        """返回默认图像"""
        return torch.zeros(3, 84, 84)

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
