import numpy as np
from torchvision import transforms


class DataAugmentation:
    """多视角一致性数据增强"""

    def __init__(self, img_size=256):
        self.img_size = img_size

    def __call__(self, imgs, depths=None):
        # 随机水平翻转（保持多视角一致性）
        if np.random.rand() > 0.5:
            imgs = transforms.functional.hflip(imgs)
            if depths is not None:
                depths = transforms.functional.hflip(depths)

        # 随机颜色抖动（单张图像应用）
        color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        for i in range(imgs.size(0)):
            imgs[i] = color_jitter(imgs[i])

        return imgs, depths
