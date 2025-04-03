import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging


class ImageFixer:
    def __init__(self, root_dir, categories=None):
        self.root_dir = Path(root_dir)
        self.categories = categories or [
            '02691156',  # airplane
            '02828884',  # bench
            '03001627',  # chair
            '03211117',  # display
            '04379243'  # table
        ]

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('image_fixing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def fix_category(self, category_id):
        """修复单个类别的所有图像"""
        category_dir = self.root_dir / category_id
        if not category_dir.exists():
            self.logger.error(f"Category directory not found: {category_dir}")
            return

        self.logger.info(f"Processing category: {category_id}")

        # 查找所有screenshots目录
        screenshot_dirs = list(category_dir.glob('*/screenshots'))

        if not screenshot_dirs:
            self.logger.warning(f"No screenshot directories found in category {category_id}")
            return

        self.logger.info(f"Found {len(screenshot_dirs)} screenshot directories in category {category_id}")

        # 处理每个目录
        for screenshots_dir in tqdm(screenshot_dirs, desc=f"Fixing images in {category_id}"):
            self.fix_screenshots_dir(screenshots_dir)

    def fix_screenshots_dir(self, screenshots_dir):
        """修复单个screenshots目录中的所有图像"""
        try:
            # 获取所有PNG文件
            png_files = list(screenshots_dir.glob('*.png'))

            if not png_files:
                self.logger.debug(f"No PNG files found in {screenshots_dir}")
                return

            # 处理每个图像
            for png_file in png_files:
                self.fix_image(png_file)

        except Exception as e:
            self.logger.error(f"Error processing directory {screenshots_dir}: {str(e)}")

    def fix_image(self, image_path):
        """修复单个图像文件"""
        try:
            # 创建临时文件路径
            temp_path = image_path.with_suffix('.tmp.png')

            # 尝试打开图像
            try:
                # 使用PIL打开图像
                img = Image.open(image_path)
                img = img.convert('RGB')  # 确保是RGB格式

                # 保存为新文件
                img.save(temp_path)

                # 如果成功，替换原文件
                os.replace(temp_path, image_path)
                return True

            except Exception as e:
                # 如果PIL打开失败，尝试使用OpenCV
                try:
                    # 使用OpenCV读取图像
                    img = cv2.imread(str(image_path))
                    if img is None:
                        raise ValueError("OpenCV couldn't read the image")

                    # 保存为新文件
                    cv2.imwrite(str(temp_path), img)

                    # 如果成功，替换原文件
                    os.replace(temp_path, image_path)
                    return True

                except Exception as cv_error:
                    self.logger.error(f"Both PIL and OpenCV failed to fix {image_path}: {str(e)}, {str(cv_error)}")
                    return False

        except Exception as e:
            self.logger.error(f"Error fixing image {image_path}: {str(e)}")
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    def fix_all_categories(self):
        """修复所有类别的图像"""
        self.logger.info(f"Starting image fixing for {len(self.categories)} categories")

        for category_id in self.categories:
            self.fix_category(category_id)

        self.logger.info("Image fixing completed")


def main():
    # 设置路径
    root_dir = "/scratch/duanyiyang/datasets/shapenet/ShapeNetCore.v2/ShapeNetCore.v2"

    # 指定要处理的类别
    categories = [
        '02691156',  # airplane
        '02828884',  # bench
        '03001627',  # chair
        '03211117',  # display
        '04379243'  # table
    ]

    fixer = ImageFixer(
        root_dir=root_dir,
        categories=categories
    )

    # 开始修复
    fixer.fix_all_categories()


if __name__ == "__main__":
    main()
