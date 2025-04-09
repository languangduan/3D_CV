# models/lighting.py
import torch
from torch import nn


class LightingModel(nn.Module):
    def __init__(self):
        super().__init__()

    def apply_lighting(self, points, normals, albedo, lighting_params):
        """
        应用光照模型

        Args:
            points: 点云坐标 [B, N, 3]
            normals: 点云法线 [B, N, 3]
            albedo: 基础颜色 [B, N, 3]
            lighting_params: 光照参数 [B, 9]
                - ambient: [B, 3] - 环境光
                - diffuse: [B, 3] - 漫反射
                - specular: [B, 3] - 镜面反射

        Returns:
            shaded: 应用光照后的颜色 [B, N, 3]
        """
        B, N, _ = points.shape

        # 解析光照参数
        ambient = lighting_params[:, :3].view(B, 1, 3)
        diffuse = lighting_params[:, 3:6].view(B, 1, 3)
        specular = lighting_params[:, 6:9].view(B, 1, 3)

        # 光源方向（假设固定）
        light_dir = torch.tensor([0.0, 0.0, 1.0], device=points.device)
        light_dir = light_dir.view(1, 1, 3).expand(B, N, 3)

        # 视角方向（假设从z轴正方向观察）
        view_dir = torch.tensor([0.0, 0.0, 1.0], device=points.device)
        view_dir = view_dir.view(1, 1, 3).expand(B, N, 3)

        # 计算漫反射项
        n_dot_l = torch.clamp(torch.sum(normals * light_dir, dim=2, keepdim=True), 0, 1)
        diffuse_term = diffuse * n_dot_l

        # 计算镜面反射项
        reflection = 2 * n_dot_l * normals - light_dir
        r_dot_v = torch.clamp(torch.sum(reflection * view_dir, dim=2, keepdim=True), 0, 1)
        specular_term = specular * torch.pow(r_dot_v, 32)

        # 组合光照项
        shaded = albedo * (ambient + diffuse_term) + specular_term

        # 裁剪到[0,1]范围
        shaded = torch.clamp(shaded, 0, 1)

        return shaded
