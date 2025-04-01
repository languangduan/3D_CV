# losses/edge_aware_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareLoss(nn.Module):
    def __init__(self, lambda_lap=0.1, beta_grad=0.05):
        super().__init__()
        self.lambda_lap = lambda_lap
        self.beta_grad = beta_grad

    def forward(self, pred_mesh, gt_mesh=None):
        """
        计算边缘感知损失
        Args:
            pred_mesh: 预测网格
            gt_mesh: 可选的真值网格
        """
        # 获取顶点和面
        verts = pred_mesh.verts_padded()
        faces = pred_mesh.faces_padded()

        # 1. 计算拉普拉斯项 (L1范数)
        lap_loss = self._compute_laplacian(verts, faces)

        # 2. 计算梯度一致性项 (如果有gt_mesh)
        grad_loss = 0.0
        if gt_mesh is not None:
            gt_verts = gt_mesh.verts_padded()
            gt_faces = gt_mesh.faces_padded()

            pred_grad = self._compute_gradient(verts, faces)
            gt_grad = self._compute_gradient(gt_verts, gt_faces)

            grad_loss = F.mse_loss(pred_grad, gt_grad)

        # 总损失
        total_loss = self.lambda_lap * lap_loss + self.beta_grad * grad_loss

        return total_loss

    def _compute_laplacian(self, verts, faces):
        """计算网格的拉普拉斯算子"""
        # 为简化示例，这里使用PyTorch3D的实现
        from pytorch3d.loss import mesh_laplacian_smoothing

        lap = mesh_laplacian_smoothing(verts, faces)
        return lap.abs().mean()  # L1范数

    def _compute_gradient(self, verts, faces):
        """计算网格表面的梯度"""
        # 获取面的三个顶点
        v0 = torch.index_select(verts, 1, faces[:, :, 0])
        v1 = torch.index_select(verts, 1, faces[:, :, 1])
        v2 = torch.index_select(verts, 1, faces[:, :, 2])

        # 计算面的法向量
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=2)
        face_normals = F.normalize(face_normals, dim=2)

        return face_normals
