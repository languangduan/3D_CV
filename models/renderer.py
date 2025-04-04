# models/renderer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.ops import cubify


class VolumeRenderer(nn.Module):
    def __init__(self, num_samples=64, ray_step_size=0.01, threshold=0.5):
        super().__init__()
        self.num_samples = num_samples
        self.ray_step_size = ray_step_size
        self.threshold = threshold

    def forward(self, points, densities):
        """
        将密度场转换为网格
        Args:
            points: 采样点 [B, N, 3]
            densities: 密度值 [B, N, 1]
        Returns:
            meshes: PyTorch3D网格对象
        """
        B = points.shape[0]

        # 将点和密度重塑为体积网格
        grid_size = int(round(points.shape[1] ** (1 / 3)))
        densities = densities.view(B, grid_size, grid_size, grid_size)

        # 使用阈值生成二值体素网格
        voxels = (densities > self.threshold).float()

        # 使用PyTorch3D的cubify将体素转换为网格
        meshes = []
        verts_list = []
        faces_list = []

        for i in range(B):
            mesh = cubify(voxels[i:i + 1], self.threshold)

            # 检查mesh是否为空
            if len(mesh.verts_list()) > 0 and len(mesh.faces_list()) > 0:
                verts = mesh.verts_list()[0]
                faces = mesh.faces_list()[0]
                verts_list.append(verts)
                faces_list.append(faces)
            else:
                # 创建一个小的默认网格（单个三角形）以避免空网格
                default_verts = torch.tensor([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0]
                ], device=points.device)
                default_faces = torch.tensor([[0, 1, 2]], device=points.device, dtype=torch.long)
                verts_list.append(default_verts)
                faces_list.append(default_faces)

        # 使用正确的构造方法创建Meshes对象
        return Meshes(verts=verts_list, faces=faces_list)
