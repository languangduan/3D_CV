import os
import numpy as np
import trimesh
import pyrender
import cv2
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
import logging
import matplotlib.pyplot as plt


class ShapeNetPreprocessor:
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

    def __init__(self, root_dir, num_views=24, image_size=224, num_workers=1, categories=None):
        self.root_dir = Path(root_dir)
        self.num_views = num_views
        self.image_size = image_size
        self.num_workers = num_workers
        self.categories = categories if categories else list(self.CATEGORY_MAP.keys())

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('preprocessing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # 设置PyRender
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

    def create_scene(self, mesh):
        """创建新的渲染场景"""
        scene = pyrender.Scene(ambient_light=np.array([0.5, 0.5, 0.5, 1.0]))

        # 添加网格
        mesh_node = scene.add(mesh)

        # 添加相机
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_node = scene.add(camera)

        # 添加光源
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        light_node = scene.add(light)

        return scene, mesh_node, camera_node, light_node

    def render_single_model(self, ply_path):
        """使用PyRender渲染单个模型的多个视角"""
        try:
            # 创建输出目录
            screenshots_dir = Path(ply_path).parent.parent / 'screenshots'
            screenshots_dir.mkdir(exist_ok=True)

            # 如果已经存在渲染文件，跳过
            existing_renders = list(screenshots_dir.glob('*.png'))
            if len(existing_renders) >= self.num_views:
                self.logger.debug(f"Skipping {ply_path}: Already rendered")
                return True

            # 加载PLY模型
            mesh = trimesh.load(ply_path)

            # 确保模型居中和归一化
            mesh.vertices -= mesh.center_mass
            scale = 1.0 / np.max(np.abs(mesh.vertices))
            mesh.vertices *= scale

            # 转换为PyRender mesh
            mesh = pyrender.Mesh.from_trimesh(mesh)

            # 创建场景和渲染器
            scene, mesh_node, camera_node, light_node = self.create_scene(mesh)
            r = pyrender.OffscreenRenderer(self.image_size, self.image_size)

            # 渲染不同视角
            for i in range(self.num_views):
                # 计算相机位置
                theta = i * (360.0 / self.num_views)
                rad = np.radians(theta)

                # 更新相机位置
                camera_pose = np.array([
                    [np.cos(rad), 0, -np.sin(rad), 2 * np.sin(rad)],
                    [0, 1, 0, 0],
                    [np.sin(rad), 0, np.cos(rad), 2 * np.cos(rad)],
                    [0, 0, 0, 1]
                ])
                scene.set_pose(camera_node, camera_pose)

                # 渲染
                color, _ = r.render(scene)

                # 保存图像
                output_path = screenshots_dir / f'{Path(ply_path).parent.parent.name}_{i:03d}.png'
                plt.imsave(str(output_path), color)

            # 清理
            r.delete()

            return True

        except Exception as e:
            self.logger.error(f"Error processing {ply_path}: {str(e)}")
            return False

    def process_category(self, category_id):
        """处理单个类别目录下的所有模型"""
        category_dir = self.root_dir / category_id
        if not category_dir.exists():
            self.logger.error(f"Category directory not found: {category_dir}")
            return

        self.logger.info(f"Processing category: {self.CATEGORY_MAP[category_id]} ({category_id})")

        # 查找所有PLY文件
        ply_files = list(category_dir.rglob('**/model_normalized.ply'))

        if not ply_files:
            self.logger.warning(f"No PLY files found in category {category_id}")
            return

        self.logger.info(f"Found {len(ply_files)} models in category {category_id}")

        # 串行处理而不是并行处理
        for ply_file in tqdm(ply_files, desc=f"Rendering {self.CATEGORY_MAP[category_id]}"):
            self.render_single_model(ply_file)

    def process_dataset(self):
        """处理指定类别的模型"""
        self.logger.info(f"Starting preprocessing for {len(self.categories)} categories")
        self.logger.info(f"Categories to process: {[self.CATEGORY_MAP[cat] for cat in self.categories]}")

        for category_id in self.categories:
            if category_id not in self.CATEGORY_MAP:
                self.logger.warning(f"Unknown category ID: {category_id}")
                continue
            self.process_category(category_id)

        self.logger.info("Dataset preprocessing completed")


def main():
    """主函数"""
    # 设置参数
    root_dir = "/scratch/duanyiyang/datasets/shapenet/ShapeNetCore.v2/ShapeNetCore.v2"

    # 指定要处理的类别
    categories = [
        '02691156',  # airplane
        '02828884',  # bench
        '03001627',  # chair
        '03211117',  # display
        '04379243'  # table
    ]

    preprocessor = ShapeNetPreprocessor(
        root_dir=root_dir,
        num_views=24,
        image_size=224,
        num_workers=1,  # 改为串行处理
        categories=categories
    )

    # 开始处理
    preprocessor.process_dataset()


if __name__ == "__main__":
    main()
