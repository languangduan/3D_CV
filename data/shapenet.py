import os
import json
import re

import numpy as np
import trimesh
from PIL import Image
import torch
from pytorch3d.structures import Meshes
from torch.utils.data import Dataset
from torchvision import transforms
import kagglehub
from pathlib import Path
import shutil
from tqdm import tqdm



class ShapeNetDataset(Dataset):
    # ShapeNet类别ID到名称的映射
    CATEGORY_MAP = {
        '02691156': 'airplane',
        '02828884': 'bench',
        '02933112': 'cabinet',
        '02958343': 'car',
        '03001627': 'chair',
        '03211117': 'display',
        '03636649': 'lamp',
        '03691459': 'speaker',
        '04090263': 'rifle',
        '04256520': 'sofa',
        '04379243': 'table',
        '04401088': 'telephone',
        '04530566': 'vessel'
    }

    # 添加类别描述，用于CLIP文本编码
    CATEGORY_DESCRIPTIONS = {
        '02691156': ['a 3D airplane', 'a model of an airplane', 'a rendering of an airplane'],
        '02828884': ['a 3D bench', 'a model of a bench', 'a rendering of a bench'],
        '02933112': ['a 3D cabinet', 'a model of a cabinet', 'a rendering of a cabinet'],
        '02958343': ['a 3D car', 'a model of a car', 'a rendering of a car'],
        '03001627': ['a 3D chair', 'a model of a chair', 'a rendering of a chair'],
        '03211117': ['a 3D display', 'a model of a display', 'a rendering of a display'],
        '03636649': ['a 3D lamp', 'a model of a lamp', 'a rendering of a lamp'],
        '03691459': ['a 3D speaker', 'a model of a speaker', 'a rendering of a speaker'],
        '04090263': ['a 3D rifle', 'a model of a rifle', 'a rendering of a rifle'],
        '04256520': ['a 3D sofa', 'a model of a sofa', 'a rendering of a sofa'],
        '04379243': ['a 3D table', 'a model of a table', 'a rendering of a table'],
        '04401088': ['a 3D telephone', 'a model of a telephone', 'a rendering of a telephone'],
        '04530566': ['a 3D vessel', 'a model of a vessel', 'a rendering of a vessel']
    }

    def __init__(self, root_dir=None, transform=None, categories=['02691156'],
                 split='train', img_size=256, use_depth=True,
                 samples_per_category=None, download=True,
                 load_meshes=True, num_points=2048):

        """
        单视角ShapeNet数据集加载器
        Args:
            root_dir: ShapeNet根目录，如果为None则使用默认路径
            categories: 类别ID列表或类别名称列表
            split: 数据集划分 (train/val/test)
            img_size: 输出图像尺寸
            use_depth: 是否加载深度图
            samples_per_category: 每个类别采样的实例数量，None表示使用全部
            download: 是否自动下载数据集
            load_meshes: 是否加载PLY网格文件
        """
        # 设置默认路径
        if root_dir is None:
            root_dir = os.path.expanduser('/scratch/duanyiyang/datasets/shapenet')
        self.root = root_dir
        self.load_meshes = load_meshes

        # 处理类别输入（支持ID或名称）
        self.categories = self._process_categories(categories)

        self.img_size = img_size
        self.use_depth = use_depth
        self.samples_per_category = samples_per_category
        self.split = split

        # 如果需要，下载数据集
        if download:
            self._download_dataset()

        # 数据增强
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # 加载所有样本和视角
        self.all_samples = []
        raw_samples = self._load_samples(split)

        # 展开样本和视角
        for sample in raw_samples:
            for img_file in sample['image_files']:
                self.all_samples.append({
                    'sample': sample,
                    'img_file': img_file
                })

        # 原有的初始化代码...
        self.num_points = num_points
        self.point_cache = {}  # 添加点云缓存

    def _process_categories(self, categories):
        """处理类别输入，支持ID和名称"""
        processed_categories = []
        for cat in categories:
            if cat in self.CATEGORY_MAP.keys():
                # 已经是类别ID
                processed_categories.append(cat)
            else:
                # 是类别名称，转换为ID
                for cat_id, cat_name in self.CATEGORY_MAP.items():
                    if cat_name == cat:
                        processed_categories.append(cat_id)
                        break
        return processed_categories

    def _load_mesh(self, model_path):
        """加载PLY模型文件"""
        if not self.load_meshes or not os.path.exists(model_path):
            print(f"Mesh file not found: {model_path}")
            return None

        try:
            # 方法1：使用trimesh加载PLY文件
            mesh = trimesh.load(model_path)
            verts = torch.tensor(mesh.vertices, dtype=torch.float32)
            faces = torch.tensor(mesh.faces, dtype=torch.long)

            # 创建PyTorch3D的Meshes对象
            pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])
            return pytorch3d_mesh

            # 方法2：直接使用PyTorch3D的IO加载PLY文件
            # pytorch3d_mesh = IO().load_mesh(model_path)
            # return pytorch3d_mesh

        except Exception as e:
            print(f"Error loading mesh from {model_path}: {e}")
            return None

    def _process_image(self, img_path):
        """处理图像文件"""
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个空白图像作为替代
            return torch.zeros(3, self.img_size, self.img_size)

    def _download_dataset(self):
        """下载并解压数据集"""
        if os.path.exists(self.root) and len(os.listdir(self.root)) > 0:
            print(f"Dataset already exists at {self.root}")
            return

        print("Downloading ShapeNet dataset...")
        try:
            # 使用kagglehub下载
            path = kagglehub.dataset_download("hajareddagni/shapenetcorev2")

            # 确保目标目录存在
            os.makedirs(self.root, exist_ok=True)

            # 移动或解压文件到目标目录
            if path.endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(self.root)
            else:
                # 如果是目录，直接复制内容
                for item in os.listdir(path):
                    s = os.path.join(path, item)
                    d = os.path.join(self.root, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)

            print(f"Dataset downloaded and extracted to {self.root}")
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            raise

    def _load_samples_(self, split):
        """加载数据样本，并限制每个实例的图像数量"""
        samples = []
        for cat_id in self.categories:
            cat_samples = []
            cat_dir = os.path.join(self.root, 'ShapeNetCore.v2', 'ShapeNetCore.v2', cat_id)

            print(f"\nDebug - Checking category directory: {cat_dir}")

            if not os.path.exists(cat_dir):
                print(f"Warning: Category directory {cat_dir} not found")
                continue

            # 直接列出类别目录下的所有实例目录
            instance_dirs = [d for d in os.listdir(cat_dir)
                             if os.path.isdir(os.path.join(cat_dir, d))]
            print(f"Debug - Found {len(instance_dirs)} instance directories")

            # 应用采样限制
            if self.samples_per_category is not None:
                if len(instance_dirs) > self.samples_per_category:
                    instance_dirs = np.random.choice(
                        instance_dirs,
                        self.samples_per_category,
                        replace=False
                    ).tolist()

            for instance_id in tqdm(instance_dirs, desc=f"Loading {self.CATEGORY_MAP[cat_id]}"):
                screenshots_dir = os.path.join(cat_dir, instance_id, 'screenshots')
                model_path = os.path.join(cat_dir, instance_id, 'models', 'model_normalized.ply')

                print(f"Checking instance: {instance_id}")
                print(f"  screenshots_dir: {screenshots_dir}")
                if not os.path.exists(screenshots_dir):
                    print(f"  Screenshots directory NOT found.")
                    continue

                all_image_files = sorted([f for f in os.listdir(screenshots_dir) if f.endswith('.png')])
                print(f"  Found {len(all_image_files)} png files.")

                r_images = [f for f in all_image_files if f.startswith('screenshots_r_') and f.endswith('.png')]
                print(f"  Found {len(r_images)} screenshots_r_*.png files.")

                if not r_images:
                    print(f"  No screenshots_r_XXX.png images, skipping sample.")
                    continue

                if len(r_images) > 6:
                    step = len(r_images) // 6
                    image_files = r_images[::step][:6]
                else:
                    image_files = r_images

                if image_files:
                    print(f"  Using {len(image_files)} images for this sample.")
                    cat_samples.append({
                        'instance_id': instance_id,
                        'category_id': cat_id,
                        'category_name': self.CATEGORY_MAP[cat_id],
                        'screenshots_dir': screenshots_dir,
                        'image_files': image_files,
                        'model_path': model_path
                    })

            # for instance_id in tqdm(instance_dirs, desc=f"Loading {self.CATEGORY_MAP[cat_id]}"):
            #     # 更新screenshots目录的路径
            #     screenshots_dir = os.path.join(cat_dir, instance_id, 'screenshots')
            #
            #     # 更新模型文件路径为PLY文件
            #     model_path = os.path.join(cat_dir, instance_id, 'models', 'model_normalized.ply')
            #
            #     if os.path.exists(screenshots_dir):
            #         # 获取所有png文件
            #         all_image_files = sorted([f for f in os.listdir(screenshots_dir)
            #                                   if f.endswith('.png')])
            #
            #         # 从24张图像中均匀选择6张
            #         if len(all_image_files) > 6:
            #             # 计算采样间隔，确保均匀采样
            #             step = len(all_image_files) // 6
            #             image_files = all_image_files[::step][:6]  # 取步长为step的6张图像
            #         else:
            #             image_files = all_image_files  # 如果图像少于6张，使用全部
            #
            #         if image_files:  # 如果有图片文件
            #             cat_samples.append({
            #                 'instance_id': instance_id,
            #                 'category_id': cat_id,
            #                 'category_name': self.CATEGORY_MAP[cat_id],
            #                 'screenshots_dir': screenshots_dir,
            #                 'image_files': image_files,  # 这里已经减少到6张或更少
            #                 'model_path': model_path
            #             })
            #     else:
            #         print(f"Debug - Screenshots directory not found: {screenshots_dir}")

            samples.extend(cat_samples)
            print(f"Loaded {len(cat_samples)} samples for category {self.CATEGORY_MAP[cat_id]}")

        if len(samples) == 0:
            raise RuntimeError("No samples found in the dataset! Please check the data directory structure.")

        return samples

    def _load_samples(self, split):
        """加载数据样本，并限制每个实例的图像数量"""
        samples = []
        pattern = re.compile(r'^screenshots_r_\d+\.png$')  # 只匹配 screenshots_r_数字.png 格式

        for cat_id in self.categories:
            cat_samples = []
            cat_dir = os.path.join(self.root, 'ShapeNetCore.v2', 'ShapeNetCore.v2', cat_id)

            print(f"\nDebug - Checking category directory: {cat_dir}")

            if not os.path.exists(cat_dir):
                print(f"Warning: Category directory {cat_dir} not found")
                continue

            # 直接列出类别目录下的所有实例目录（顺序遍历）
            instance_dirs = [d for d in sorted(os.listdir(cat_dir))
                             if os.path.isdir(os.path.join(cat_dir, d))]
            print(f"Debug - Found {len(instance_dirs)} instance directories")

            # 目标样本数
            target_num = self.samples_per_category if self.samples_per_category is not None else len(instance_dirs)

            for instance_id in tqdm(instance_dirs, desc=f"Loading {self.CATEGORY_MAP[cat_id]}"):
                if len(cat_samples) >= target_num:
                    # 已经找到足够样本，停止继续遍历
                    break

                screenshots_dir = os.path.join(cat_dir, instance_id,) # 'screenshots')
                model_path = os.path.join(cat_dir, instance_id, 'models', 'model_normalized.ply')

                print(f"Checking instance: {instance_id}")
                print(f"  screenshots_dir: {screenshots_dir}")
                if not os.path.exists(screenshots_dir):
                    print(f"  Screenshots directory NOT found.")
                    continue

                all_image_files = sorted([f for f in os.listdir(screenshots_dir) if f.endswith('.png')])
                print(f"  Found {len(all_image_files)} png files.")

                # 只保留严格匹配 screenshots_r_数字.png 格式的图片
                r_images = [f for f in all_image_files if pattern.match(f)]
                print(f"  Found {len(r_images)} screenshots_r_XXX.png files.")

                if not r_images:
                    print(f"  No screenshots_r_XXX.png images, skipping sample.")
                    continue

                # 使用所有符合条件的图片（不再采样）
                image_files = r_images

                if image_files:
                    print(f"  Using {len(image_files)} images for this sample.")
                    cat_samples.append({
                        'instance_id': instance_id,
                        'category_id': cat_id,
                        'category_name': self.CATEGORY_MAP[cat_id],
                        'screenshots_dir': screenshots_dir,
                        'image_files': image_files,
                        'model_path': model_path
                    })

            samples.extend(cat_samples)
            print(f"Loaded {len(cat_samples)} samples for category {self.CATEGORY_MAP[cat_id]}")

        if len(samples) == 0:
            raise RuntimeError("No samples found in the dataset! Please check the data directory structure.")

        return samples
    def __len__(self):
        """返回数据集大小"""
        return len(self.all_samples)

    def __getitem__(self, idx):
        """获取单个视角的样本"""
        item = self.all_samples[idx]
        sample = item['sample']
        img_file = item['img_file']
        model_path = sample['model_path']

        # 加载RGB图像
        img_path = os.path.join(sample['screenshots_dir'], img_file)
        img = self._process_image(img_path)

        # 加载或生成点云
        if model_path in self.point_cache:
            points = self.point_cache[model_path]
        else:
            try:
                mesh = trimesh.load(model_path)
                points = self._sample_points_from_mesh(mesh, self.num_points)
                # 缓存点云数据
                self.point_cache[model_path] = points
            except Exception as e:
                print(f"Error loading mesh from {model_path}: {e}")
                points = torch.zeros((self.num_points, 3), dtype=torch.float32)

        # 构建返回字典
        data_dict = {
            'image': img,
            'points': points,  # 添加点云数据
            'category': sample['category_id'],
            'category_name': sample['category_name'],
            'instance_id': sample['instance_id'],
            'text_prompts': np.random.choice(self.CATEGORY_DESCRIPTIONS[sample['category_id']]),
            'model_path': model_path,
            'depth': torch.ones(1, self.img_size, self.img_size) * 0.5 if self.use_depth else torch.zeros(1,
                                                                                                          self.img_size,
                                                                                                          self.img_size)
        }

        return data_dict

    def clear_cache(self):
        """清理点云缓存"""
        self.point_cache.clear()

    def _load_depth(self, path):
        """加载深度图并归一化到[0,1]"""
        depth = Image.open(path)
        depth = np.array(depth).astype(np.float32) / 65535.0  # 16bit转0-1
        depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]

        # 应用相同的数据增强
        depth = transforms.functional.resize(depth, self.img_size)
        depth = transforms.functional.center_crop(depth, self.img_size)
        return depth

    def _load_camera(self, render_dir, prefix):
        """加载相机内外参数"""
        # 加载外参
        pose_path = os.path.join(render_dir, f'{prefix}_pose.txt')
        ext = np.loadtxt(pose_path)  # [4,4]

        # 加载内参
        intrinsic_path = os.path.join(render_dir, f'{prefix}_intrinsic.txt')
        K = np.loadtxt(intrinsic_path)[:3, :3]  # 3x3

        # 合成完整投影矩阵
        proj_matrix = K @ ext
        return proj_matrix

    def _sample_points_from_mesh(self, mesh, num_points=2048):
        try:
            points = mesh.sample(num_points)
            points = torch.tensor(points, dtype=torch.float32)

            # 使用bounding box归一化
            bbox = mesh.bounding_box.bounds.astype(np.float32)
            center = torch.from_numpy((bbox[0] + bbox[1]) / 2).to(points.device).float()
            scale = float(((bbox[1] - bbox[0]).max() / 2))
            scale = max(scale, 1e-6)
            points = (points - center) / scale

            assert not torch.isnan(points).any(), "NaN detected in points"
            assert not torch.isinf(points).any(), "Inf detected in points"

            return points
        except Exception as e:
            print(f"Error sampling points: {e}")
            return torch.randn((num_points, 3), dtype=torch.float32) * 0.1

    def get_category_statistics(self):
        """获取数据集类别统计信息"""
        stats = {}
        for sample in self.samples:
            cat_name = sample['category_name']
            if cat_name not in stats:
                stats[cat_name] = 0
            stats[cat_name] += 1
        return stats

# 使用示例
def test_dataset():
    # 通过类别名称创建数据集
    dataset = ShapeNetDataset(
        categories=['airplane', 'chair'],  # 使用类别名称
        samples_per_category=10,  # 每个类别只取10个样本
        img_size=128,  # 降低图像分辨率
        num_views=4,   # 每个物体只取4个视角
        download=True  # 自动下载数据集
    )

    # 打印数据集统计信息
    print("\nDataset Statistics:")
    stats = dataset.get_category_statistics()
    for cat_name, count in stats.items():
        print(f"{cat_name}: {count} samples")

    # 测试加载一个样本
    sample = dataset[0]
    print("\nSample content:")
    for key, value in sample.items():
        if torch.is_tensor(value):
            print(f"{key}: tensor shape {value.shape}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    test_dataset()
