import os
import json
import numpy as np
from PIL import Image
import torch
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

    def __init__(self, root_dir=None, categories=['02691156'], split='train',
                 img_size=256, num_views=24, use_depth=True,
                 samples_per_category=None, download=True):
        """
        增强版ShapeNet数据集加载器
        Args:
            root_dir: ShapeNet根目录，如果为None则使用默认路径
            categories: 类别ID列表或类别名称列表
            split: 数据集划分 (train/val/test)
            img_size: 输出图像尺寸
            num_views: 每个物体加载的视角数
            use_depth: 是否加载深度图
            samples_per_category: 每个类别采样的实例数量，None表示使用全部
            download: 是否自动下载数据集
        """
        # 设置默认路径
        if root_dir is None:
            root_dir = os.path.expanduser('~/datasets/shapenet')
        self.root = root_dir

        # 处理类别输入（支持ID或名称）
        self.categories = self._process_categories(categories)

        self.img_size = img_size
        self.num_views = num_views
        self.use_depth = use_depth
        self.samples_per_category = samples_per_category

        # 如果需要，下载数据集
        if download:
            self._download_dataset()

        # 数据增强
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 加载数据路径
        self.samples = self._load_samples(split)

    def _process_categories(self, categories):
        """处理类别输入，支持ID或名称"""
        processed_categories = []
        for cat in categories:
            if cat in self.CATEGORY_MAP:  # 是类别ID
                processed_categories.append(cat)
            else:  # 假设是类别名称
                for id_, name in self.CATEGORY_MAP.items():
                    if name.lower() == cat.lower():
                        processed_categories.append(id_)
                        break
        if not processed_categories:
            raise ValueError(f"No valid categories found in {categories}")
        return processed_categories

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

    def _load_samples(self, split):
        """加载数据样本，适配新的目录结构"""
        samples = []
        for cat_id in self.categories:
            cat_samples = []
            cat_dir = os.path.join(self.root, 'ShapeNetCore.v2','ShapeNetCore.v2', cat_id)

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
                # 更新screenshots目录的路径
                screenshots_dir = os.path.join(cat_dir, instance_id, 'screenshots')
                if os.path.exists(screenshots_dir):
                    # 获取所有png文件
                    image_files = sorted([f for f in os.listdir(screenshots_dir)
                                          if f.endswith('.png')])

                    if image_files:  # 如果有图片文件
                        cat_samples.append({
                            'instance_id': instance_id,
                            'category_id': cat_id,
                            'category_name': self.CATEGORY_MAP[cat_id],
                            'screenshots_dir': screenshots_dir,
                            'image_files': image_files,
                            'model_path': os.path.join(cat_dir, instance_id, 'models/model.obj')
                        })
                else:
                    print(f"Debug - Screenshots directory not found: {screenshots_dir}")

            samples.extend(cat_samples)
            print(f"Loaded {len(cat_samples)} samples for category {self.CATEGORY_MAP[cat_id]}")

        if len(samples) == 0:
            raise RuntimeError("No samples found in the dataset! Please check the data directory structure.")

        return samples

    def __getitem__(self, idx):
        """更新数据加载逻辑以匹配新的目录结构"""
        sample = self.samples[idx]

        # 随机选择视角
        num_views = min(self.num_views, len(sample['image_files']))
        selected_files = np.random.choice(sample['image_files'], num_views, replace=False)

        images = []
        for img_file in selected_files:
            # 加载RGB图像
            img_path = os.path.join(sample['screenshots_dir'], img_file)
            img = self._process_image(img_path)
            images.append(img)

        data_dict = {
            'images': torch.stack(images),
            'category': sample['category_id'],
            'category_name': sample['category_name'],
            'instance_id': sample['instance_id']
        }

        return data_dict

    def __len__(self):
        return len(self.samples)


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
