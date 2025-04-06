# losses/edge_aware_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareLoss(nn.Module):
    def __init__(self, lambda_lap=0.1, beta_grad=0.05):
        super().__init__()
        self.lambda_lap = lambda_lap
        self.beta_grad = beta_grad

    def forward(self, points, densities, depth_gradients=None):
        """
        计算边缘感知损失 - 基于点云和密度场

        Args:
            points: 采样点 [B, N, 3]
            densities: 密度值 [B, N, 1]
            depth_gradients: 从深度图计算的表面法线 [B, N, 3] 或 None

        Returns:
            edge_loss: 边缘感知损失
            metrics: 包含各损失组件的字典
        """
        # 确保points需要梯度
        if not points.requires_grad:
            points.requires_grad_(True)

        # 1. 计算密度场梯度 (∇S(x))
        grad_outputs = torch.ones_like(densities)
        sdf_grad = torch.autograd.grad(
            outputs=densities,
            inputs=points,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 2. 计算拉普拉斯项 (∇²S(x))
        laplacian = 0
        for i in range(3):  # xyz三个维度
            grad_comp = sdf_grad[:, :, i:i + 1]
            grad_outputs = torch.ones_like(grad_comp)
            second_grad = torch.autograd.grad(
                outputs=grad_comp,
                inputs=points,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            laplacian += second_grad[:, :, i:i + 1]

        # L1范数拉普拉斯损失
        laplacian_loss = torch.abs(laplacian).mean()

        # 3. 梯度一致性损失 (如果有深度梯度)
        gradient_consistency = torch.tensor(0.0, device=points.device)
        if depth_gradients is not None:
            # 归一化梯度
            sdf_grad_norm = F.normalize(sdf_grad, dim=2)
            gradient_consistency = F.mse_loss(sdf_grad_norm, depth_gradients)

        # 总损失
        edge_loss = self.lambda_lap * laplacian_loss + self.beta_grad * gradient_consistency

        return edge_loss, {
            'laplacian_loss': laplacian_loss.item(),
            'gradient_consistency': gradient_consistency.item() if isinstance(gradient_consistency,
                                                                              torch.Tensor) else gradient_consistency
        }

    # 保留旧方法以便向后兼容
    def compute_mesh_edge_loss(self, pred_mesh, gt_mesh=None):
        """
        计算基于网格的边缘感知损失（原始实现）
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
