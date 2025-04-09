# utils/depth_estimator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
class DepthEstimator(nn.Module):
    def __init__(self, model_path="checkpoints/depth_anything_v2_vitb.pth"):
        super().__init__()

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Please download it first.")

        try:
            # 导入Depth Anything V2模型


            # 创建模型实例
            self.model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])

            # 加载预训练权重
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()

            # 如果有GPU，将模型移到GPU
            if torch.cuda.is_available():
                self.model = self.model.cuda()

            print(f"Successfully loaded Depth Anything V2 model from {model_path}")
            self.model_loaded = True

        except Exception as e:
            print(f"Failed to load Depth Anything V2: {e}")
            print("Falling back to simple depth estimation")
            self.model_loaded = False

    def forward(self, images):
        """
        估计图像的深度图

        Args:
            images: 输入图像 [B, 3, H, W]，值范围[0,1]

        Returns:
            depth: 深度图 [B, 1, H, W]
            normals: 表面法线 [B, 3, H, W]
        """
        B, C, H, W = images.shape
        device = images.device

        if not self.model_loaded:
            # 如果模型未加载，返回随机深度
            depth = torch.rand(B, 1, H, W, device=device)
            normals = torch.randn(B, 3, H, W, device=device)
            normals = F.normalize(normals, dim=1)
            return depth, normals

        # 处理每张图像
        depths = []
        with torch.no_grad():
            for i in range(B):
                # 转换为numpy数组并调整格式
                img = images[i].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
                img = (img * 255).astype(np.uint8)  # 转换为0-255范围

                # BGR转换（OpenCV默认是BGR格式）
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # 推理
                depth = self.model.infer_image(img)  # [H, W]

                # 归一化深度
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

                # 转换为torch张量并添加批次和通道维度
                depth_tensor = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                depths.append(depth_tensor)

            # 拼接批次结果
            depths = torch.cat(depths, dim=0).to(device)

        # 计算表面法线
        normals = self.compute_normals_from_depth(depths)

        return depths, normals

    def compute_normals_from_depth(self, depth):
        """从深度图计算表面法线"""
        # 计算深度梯度
        grad_x = torch.zeros_like(depth)
        grad_y = torch.zeros_like(depth)

        # 使用Sobel算子计算梯度
        grad_x[:, :, :, 1:-1] = (depth[:, :, :, 2:] - depth[:, :, :, :-2]) / 2
        grad_y[:, :, 1:-1, :] = (depth[:, :, 2:, :] - depth[:, :, :-2, :]) / 2

        # 构建法线向量 [-grad_x, -grad_y, 1]
        normals = torch.cat([-grad_x, -grad_y, torch.ones_like(depth)], dim=1)

        # 归一化法线
        normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-10)

        return normals
