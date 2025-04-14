# train.py
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim
import wandb
from pytorch3d.structures import Meshes
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, default_collate
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from models.base_model import SingleViewReconstructor
from models.clip_encoder import CLIPEncoder
from losses.combined_loss import CombinedLoss
from data.shapenet import ShapeNetDataset
from utils.aug import DataAugmentation
from utils.metrics import Metrics



def mesh_collate_fn(batch):
    """
    自定义collate函数，处理包含Meshes对象的批次
    """
    # 提取所有非网格数据
    elem = batch[0]
    collated_batch = {}

    for key in elem:
        if key == 'mesh':
            # 对于网格，我们单独处理
            meshes = [item['mesh'] for item in batch]
            # 过滤掉None值
            valid_meshes = [m for m in meshes if m is not None]

            if valid_meshes:
                # 如果有有效的网格，合并它们
                if len(valid_meshes) == len(meshes):
                    # 所有网格都有效，直接合并
                    verts_list = []
                    faces_list = []
                    for mesh in valid_meshes:
                        verts_list.append(mesh.verts_padded()[0])
                        faces_list.append(mesh.faces_padded()[0])

                    collated_batch[key] = Meshes(verts=verts_list, faces=faces_list)
                else:
                    # 有些网格无效，使用None占位
                    collated_batch[key] = None
            else:
                # 所有网格都无效
                collated_batch[key] = None
        else:
            # 对于其他数据，使用默认的collate函数
            try:
                collated_batch[key] = default_collate([item[key] for item in batch])
            except TypeError:
                # 如果默认collate失败，则保持原样（列表形式）
                collated_batch[key] = [item[key] for item in batch]

    return collated_batch


class CLIPNeRFTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # 初始化模型
        self.model = SingleViewReconstructor(cfg)
        self.clip_encoder = CLIPEncoder(cfg.clip_model)

        # 损失函数
        self.loss_fn = CombinedLoss(cfg)

        # 添加渲染器配置
        self.render_size = getattr(cfg, 'render_size', 128)  # 保持128x128的渲染尺寸
        self.renderer = self.setup_renderer()


        # 添加损失历史记录
        self.loss_history = {
            'total': [],
            'chamfer': [],
            'density': [],
            'edge': [],
            'clip': [],
            'depth': [],
            'reg': []
        }

        # 设置保存路径
        self.loss_log_path = "loss_history.json"


    def on_train_epoch_end(self):
        """在每个训练epoch结束时保存损失历史"""
        import json
        import os

        # 保存当前的损失历史
        with open(self.loss_log_path, 'w') as f:
            json.dump(self.loss_history, f)

        # 如果使用wandb，也可以记录为artifact
        if self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log_artifact(self.loss_log_path, name="loss_history", type="json")

    def setup_renderer(self):
        """设置PyTorch3D渲染器"""
        from pytorch3d.renderer import (
            look_at_view_transform,
            FoVPerspectiveCameras,
            PointLights,
            RasterizationSettings,
            MeshRenderer,
            MeshRasterizer,
            SoftPhongShader
        )

        # 设置相机
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        # 设置光照
        lights = PointLights(
            device=self.device,
            location=[[0, 0, 3]],
            ambient_color=[[0.4, 0.4, 0.4]],
            diffuse_color=[[0.6, 0.6, 0.6]],
            specular_color=[[0.3, 0.3, 0.3]]
        )

        # 设置光栅化器
        raster_settings = RasterizationSettings(
            image_size=self.render_size,
            blur_radius=0.0,
            faces_per_pixel=1
        )

        # 设置着色器
        shader = SoftPhongShader(
            device=self.device,
            cameras=cameras,
            lights=lights
        )

        # 创建渲染器
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=shader
        )

        return renderer

    def on_fit_start(self):
        super().on_fit_start()
        self.renderer = self.setup_renderer()

        # 同步所有组件到当前设备
        self._sync_device()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.train()
        images = batch['image']
        target_points = batch['points']
        text_prompts = batch.get('text_prompts', None)

        try:
            # 前向传播
            features_list = self.model.extract_features(images)
            # print(f"Step 1: Features extracted, shapes: {[f.shape for f in features_list]}")

            # 检查特征是否有NaN
            # for i, feat in enumerate(features_list):
            #     if torch.isnan(feat).any():
            #         print(f"NaN detected in features_list[{i}]")
            #         return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

            fused_features, feature_weights, lighting_params = self.model.implicit_field.feature_pyramid(features_list)
            # print(f"Step 2: Features fused, shape: {fused_features.shape}")

            # 检查融合特征
            # if torch.isnan(fused_features).any():
            #     print(f"NaN detected in fused_features")
            #     return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

            # 创建点云
            batch_size = images.shape[0]
            points = self.model.implicit_field._create_grid_points(batch_size, images.device)
            points.requires_grad_(True)
            # print(f"Step 3: Grid points created, shape: {points.shape}")

            # 计算密度和特征
            try:
                densities, features = self.model.implicit_field.compute_density_and_features(points, fused_features)
                # print(f"Step 4: Densities and features computed, shapes: {densities.shape}, {features.shape}")

                # 检查密度和特征
                # if torch.isnan(densities).any():
                #     print(f"NaN detected in densities")
                #     return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}
                # if torch.isnan(features).any():
                #     print(f"NaN detected in features")
                #     return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

                # 计算颜色
                albedo = self.model.implicit_field.color_net(features)
                # print(f"Step 5: Albedo computed, shape: {albedo.shape}")

                # 检查albedo
                # if torch.isnan(albedo).any():
                #     print(f"NaN detected in albedo")
                #     return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

                # 计算法线
                normals = self.model.implicit_field.compute_normals(points, densities)
                # print(f"Step 6: Normals computed, shape: {normals.shape}")

                # 检查法线
                # if torch.isnan(normals).any():
                #     print(f"NaN detected in normals")
                #     return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

                # 应用光照
                colors = self.model.implicit_field.lighting_model.apply_lighting(
                    points, normals, albedo, lighting_params
                )
                # print(f"Step 7: Colors computed, shape: {colors.shape}")

                # 检查颜色
                # if torch.isnan(colors).any():
                #     print(f"NaN detected in colors")
                #     return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

                # 提取CLIP特征
                clip_features = None
                if hasattr(self, 'clip_encoder'):
                    clip_features = self.clip_encoder(images)
                    # print(f"Step 8: CLIP features extracted, shape: {clip_features.shape}")

                    # 检查CLIP特征
                    # if torch.isnan(clip_features).any():
                    #     print(f"NaN detected in clip_features")
                    #     return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

                # 提取密度场特征
                density_features = self.model.implicit_field.extract_density_features(
                    points, densities, fused_features, clip_features
                )
                # print(f"Step 9: Density features extracted, shape: {density_features.shape}")

                # 检查密度特征
                # if torch.isnan(density_features).any():
                #     print(f"NaN detected in density_features")
                #     return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

                # 计算损失
                try:
                    loss_dict = self.loss_fn({
                        'pred_points': points,
                        'pred_densities': densities,
                        'pred_colors': colors,
                        'pred_normals': normals,
                        'density_features': density_features,
                        'original_images': images,
                    }, {
                        'target_points': target_points,
                        'text_prompts': text_prompts,
                    }, model=self.model)
                    # print(f"Step 10: Loss computed: {loss_dict}")

                    # 检查各个损失组件
                    # for key, value in loss_dict.items():
                    #     if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                    #         print(f"NaN detected in loss component: {key}")
                    #         return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

                    try:
                        # 记录损失组件
                        for key, value in loss_dict.items():
                            if isinstance(value, torch.Tensor):
                                value_item = value.item()
                                self.log(f"train_{key}", value_item, prog_bar=True)

                                # 保存到历史记录
                                if key in self.loss_history:
                                    self.loss_history[key].append(value_item)
                            else:
                                self.log(f"train_{key}", value, prog_bar=True)

                                # 保存到历史记录
                                if key in self.loss_history:
                                    self.loss_history[key].append(value)
                    except Exception as e:
                        print(f"Error logging loss: {e}")

                    # 记录损失组件
                    for key, value in loss_dict.items():
                        if isinstance(value, torch.Tensor):
                            self.log(f"train_{key}", value.item(), prog_bar=True)
                        else:
                            self.log(f"train_{key}", value, prog_bar=True)

                    # 确保损失是标量且需要梯度
                    if isinstance(loss_dict['total'], torch.Tensor):
                        loss = loss_dict['total']
                    else:
                        loss = torch.tensor(loss_dict['total'], device=self.device, requires_grad=True)

                    # 最终检查
                    if torch.isnan(loss).any():
                        print(f"NaN detected in final loss")
                        return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

                    return {"loss": loss}

                except Exception as e:
                    print(f"Error in loss calculation: {e}")
                    import traceback
                    traceback.print_exc()
                    return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

            except Exception as e:
                print(f"Error in density/feature computation: {e}")
                import traceback
                traceback.print_exc()
                return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

        except Exception as e:
            print(f"Error in training step: {e}")
            import traceback
            traceback.print_exc()
            return {"loss": torch.tensor(1.0, device=self.device, requires_grad=True)}

    def _save_visualization(self, rendered_images, batch_idx):
        """保存可视化结果"""
        import torchvision

        # 选择前8个样本（或更少）
        num_samples = min(8, rendered_images.shape[0])
        images_to_save = rendered_images[:num_samples]

        # 创建网格图像
        grid = torchvision.utils.make_grid(images_to_save, nrow=4)

        # 转换为PIL图像
        from torchvision.transforms import ToPILImage
        pil_image = ToPILImage()(grid.cpu())

        # 保存图像
        import os
        os.makedirs("visualizations", exist_ok=True)
        pil_image.save(f"visualizations/epoch_{self.current_epoch}_batch_{batch_idx}.png")

    def test_step(self, batch, batch_idx):
        """测试步骤"""
        return self.validation_step(batch, batch_idx)

    def _render_mesh(self, mesh):
        """
        渲染网格为RGB图像
        Args:
            mesh: PyTorch3D Meshes对象或生成的网格
        Returns:
            渲染的图像张量 [B, 3, H, W]
        """
        batch_size = mesh.verts_padded().shape[0]
        device = mesh.device

        # 确保网格是有效的
        if not isinstance(mesh, Meshes):
            raise TypeError("Expected mesh to be a PyTorch3D Meshes object")

        # 检查网格是否为空
        if mesh.isempty():
            print("Warning: Empty mesh detected, returning blank images")
            return torch.zeros(batch_size, 3, self.render_size, self.render_size, device=device)

        # 检查批次大小是否一致
        if mesh.verts_padded().shape[0] != batch_size:
            print(
                f"Warning: Mesh batch size ({mesh.verts_padded().shape[0]}) doesn't match expected batch size ({batch_size})")
            # 尝试修复批次大小不匹配问题
            verts_list = mesh.verts_list()
            faces_list = mesh.faces_list()

            # 确保列表长度一致
            if len(verts_list) < batch_size:
                # 复制最后一个网格以匹配批次大小
                while len(verts_list) < batch_size:
                    verts_list.append(verts_list[-1])
                    faces_list.append(faces_list[-1])
            elif len(verts_list) > batch_size:
                # 截断列表以匹配批次大小
                verts_list = verts_list[:batch_size]
                faces_list = faces_list[:batch_size]

            # 重新创建Meshes对象
            mesh = Meshes(verts=verts_list, faces=faces_list)

        try:
            # 确保渲染器在正确的设备上
            if self.renderer.rasterizer.cameras.device != device:
                self.renderer = self.setup_renderer()

            # 尝试渲染网格
            images = self.renderer(mesh)
            # 提取RGB通道（丢弃alpha通道）
            images = images[..., :3].permute(0, 3, 1, 2)  # [B, 3, H, W]
            return images
        except Exception as e:
            print(f"Error during rendering: {e}")
            # 返回空白图像作为后备
            return torch.zeros(batch_size, 3, self.render_size, self.render_size, device=device)

    def _render_mesh_safe(self, meshes):
        """安全地渲染网格，处理可能的错误"""
        try:
            # 检查网格有效性
            verts = meshes.verts_padded()
            faces = meshes.faces_padded()

            # 检查是否有NaN或Inf
            if torch.isnan(verts).any() or torch.isinf(verts).any():
                print("检测到顶点中有NaN或Inf，尝试修复")
                verts = torch.where(torch.isnan(verts) | torch.isinf(verts),
                                    torch.zeros_like(verts),
                                    verts)
                # 更新网格
                from pytorch3d.structures import Meshes
                textures = meshes.textures
                meshes = Meshes(verts=verts.unbind(), faces=faces.unbind(), textures=textures)

            # 设置渲染器
            from pytorch3d.renderer import (
                look_at_view_transform, FoVPerspectiveCameras,
                RasterizationSettings, MeshRenderer, MeshRasterizer,
                SoftPhongShader
            )

            # 创建相机
            R, T = look_at_view_transform(2.0, 0, 0)
            cameras = FoVPerspectiveCameras(R=R, T=T, device=self.device)

            # 创建光照
            from pytorch3d.renderer import PointLights
            lights = PointLights(
                location=[[0.0, 0.0, 2.0]],
                ambient_color=[[0.5, 0.5, 0.5]],
                diffuse_color=[[0.3, 0.3, 0.3]],
                specular_color=[[0.2, 0.2, 0.2]],
                device=self.device
            )

            # 创建渲染器
            raster_settings = RasterizationSettings(
                image_size=256,
                blur_radius=0.0,
                faces_per_pixel=1,
            )

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                shader=SoftPhongShader(cameras=cameras, lights=lights)
            )

            # 渲染网格
            rendered_images = renderer(meshes)

            return rendered_images

        except Exception as e:
            print(f"渲染过程出错: {e}")
            import traceback
            traceback.print_exc()

            # 返回空白图像作为后备
            batch_size = len(meshes)
            return torch.ones(batch_size, 256, 256, 4, device=self.device)  # RGBA格式


    def _render_depth(self, mesh):
        """渲染网格为深度图"""
        # 使用PyTorch3D的深度渲染器
        from pytorch3d.renderer import (
            look_at_view_transform,
            FoVPerspectiveCameras,
            PointLights,
            RasterizationSettings,
            MeshRasterizer
        )

        # 设置相机
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = FoVPerspectiveCameras(device=mesh.device, R=R, T=T)

        # 设置渲染器
        raster_settings = RasterizationSettings(
            image_size=128,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # 创建光栅化器
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        # 渲染深度
        fragments = rasterizer(mesh)
        depth = fragments.zbuf

        return depth

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        # 在优化器配置中设置梯度裁剪，而不是在 training_step 中
        for group in optimizer.param_groups:
            group['max_norm'] = 1.0
            group['norm_type'] = 2.0

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_total",
                "frequency": 1
            }
        }

    def train_dataloader(self):
        # 创建ShapeNet数据集
        dataset = ShapeNetDataset(
            root_dir=self.cfg.data_root,
            split='train',
            categories=self.cfg.categories,
            transform=self.cfg.transform,
            samples_per_category=self.cfg.samples_per_category
        )

        # 创建数据加载器
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=mesh_collate_fn
        )

        return loader

    def val_dataloader(self):
        # 创建ShapeNet数据集
        dataset = ShapeNetDataset(
            root_dir=self.cfg.data_root,
            split='val',  # 使用验证集
            categories=self.cfg.categories,
            transform=self.cfg.transform,
            samples_per_category=50  # 限制验证样本数
        )

        # 创建数据加载器
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,  # 验证不需要打乱
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=mesh_collate_fn
        )

        return loader

    def generate_mesh(self, images, is_training=None):
        """
        从图像生成3D表示 - 修复了梯度问题
        """
        if is_training is None:
            is_training = self.training

        # 保存当前梯度状态
        grad_enabled = torch.is_grad_enabled()

        # 确保在整个过程中启用梯度计算
        with torch.enable_grad():
            try:
                # 保存图像引用以便后续使用
                self.last_images = images

                # 获取多尺度特征
                features_list = self.model.extract_features(images)

                # 融合特征
                fused_features, feature_weights, lighting_params = self.model.implicit_field.feature_pyramid(
                    features_list)

                # 提取CLIP特征
                clip_features = None
                if hasattr(self, 'clip_encoder'):
                    clip_features = self.clip_encoder(images)

                # 创建点云并确保需要梯度
                batch_size = images.shape[0]
                points = self.model.implicit_field._create_grid_points(batch_size, images.device)

                # 始终确保点云需要梯度，无论是否在训练模式
                points_with_grad = points.detach().clone().requires_grad_(True)

                # 计算密度和颜色
                densities, features = self.model.implicit_field.compute_density_and_features(points_with_grad,
                                                                                             fused_features)

                # 检查密度中是否有NaN
                if torch.isnan(densities).any():
                    print("检测到密度中有NaN，进行修复")
                    densities = torch.where(torch.isnan(densities),
                                            torch.ones_like(densities) * 1e-6,
                                            densities)

                albedo = self.model.implicit_field.color_net(features)

                # 确保法线计算有梯度 - 这里是关键修复点
                try:
                    # 确保点云需要梯度
                    if not points_with_grad.requires_grad:
                        points_with_grad = points_with_grad.detach().clone().requires_grad_(True)

                    # 重新计算密度以确保梯度链接
                    if not densities.requires_grad:
                        densities, _ = self.model.implicit_field.compute_density_and_features(
                            points_with_grad, fused_features)

                    # 计算法线
                    normals = self.model.implicit_field.compute_normals(points_with_grad, densities)

                    # 检查法线是否有效
                    if torch.isnan(normals).any() or torch.isinf(normals).any():
                        raise ValueError("法线中检测到NaN或Inf值")

                except Exception as e:
                    print(f"计算法线时出错: {e}")
                    print("尝试手动计算梯度...")

                    # 手动计算梯度
                    grad_outputs = torch.ones_like(densities)
                    gradients = torch.autograd.grad(
                        outputs=densities,
                        inputs=points_with_grad,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True
                    )[0]

                    # 归一化梯度得到法线
                    normals = torch.nn.functional.normalize(gradients, dim=2)

                    # 检查法线是否有效
                    if torch.isnan(normals).any() or torch.isinf(normals).any():
                        print("手动计算的法线仍有问题，使用随机法线")
                        normals = torch.nn.functional.normalize(torch.randn_like(points_with_grad), dim=2)

                # 应用光照
                colors = self.model.implicit_field.lighting_model.apply_lighting(
                    points_with_grad, normals, albedo, lighting_params
                )

                if is_training:
                    return points_with_grad, densities, colors, normals
                else:
                    # 验证/测试时生成完整网格 - 不禁用梯度
                    try:
                        verts, faces = self.convert_to_mesh(points_with_grad, densities)

                        # 验证生成的网格
                        if len(verts) == 0 or len(faces) == 0:
                            raise ValueError("生成的网格为空")

                        # 添加顶点颜色
                        verts_colors = self._interpolate_colors(points_with_grad, verts, colors)

                        # 使用TexturesVertex创建带纹理的网格
                        from pytorch3d.renderer import TexturesVertex
                        textures = TexturesVertex(verts_features=[vc.clone() for vc in verts_colors])

                        # 创建网格
                        meshes = Meshes(
                            verts=[v for v in verts],
                            faces=[f for f in faces],
                            textures=textures
                        )

                        # 验证网格有效性
                        if torch.isnan(meshes.verts_padded()).any() or torch.isinf(meshes.verts_padded()).any():
                            print("网格顶点中存在NaN或Inf，尝试修复")
                            fixed_verts = []
                            for v in verts:
                                fixed_v = torch.where(torch.isnan(v) | torch.isinf(v),
                                                      torch.zeros_like(v),
                                                      v)
                                fixed_verts.append(fixed_v)

                            meshes = Meshes(
                                verts=fixed_verts,
                                faces=[f for f in faces],
                                textures=textures
                            )

                        return meshes

                    except Exception as e:
                        print(f"网格生成失败: {e}")
                        import traceback
                        traceback.print_exc()

                        # 创建一个简单的立方体作为后备
                        print("使用默认立方体作为后备")
                        default_meshes = []

                        for b in range(batch_size):
                            verts = torch.tensor([
                                [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                                [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
                            ], dtype=torch.float32, device=images.device)

                            faces = torch.tensor([
                                [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                                [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                                [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]
                            ], dtype=torch.int64, device=images.device)

                            # 创建简单的顶点颜色
                            verts_rgb = torch.ones_like(verts)  # 白色

                            default_meshes.append({
                                'verts': verts,
                                'faces': faces,
                                'verts_rgb': verts_rgb
                            })

                        # 创建带纹理的网格
                        from pytorch3d.renderer import TexturesVertex


                        textures = TexturesVertex(verts_features=[m['verts_rgb'] for m in default_meshes])

                        return Meshes(
                            verts=[m['verts'] for m in default_meshes],
                            faces=[m['faces'] for m in default_meshes],
                            textures=textures
                        )

            except Exception as e:
                print(f"generate_mesh中发生未处理的错误: {e}")
                import traceback
                traceback.print_exc()

                # 返回一个空的网格作为最后的后备
                from pytorch3d.structures import Meshes
                return Meshes(verts=[], faces=[])

            finally:
                # 恢复原始梯度状态
                torch.set_grad_enabled(grad_enabled)

    def _interpolate_colors(self, points, verts_list, colors):
        """插值点云颜色到网格顶点 - 修复版本"""
        try:
            batch_size = points.shape[0]
            verts_colors = []

            for b in range(batch_size):
                # 获取单个样本的点云、顶点和颜色
                sample_points = points[b].detach()
                sample_verts = verts_list[b]
                sample_colors = colors[b].detach()

                # 构建KD树进行最近邻查询
                from sklearn.neighbors import KDTree
                points_np = sample_points.cpu().numpy()
                tree = KDTree(points_np)

                # 查询每个顶点的最近邻点
                verts_np = sample_verts.cpu().numpy()
                _, indices = tree.query(verts_np, k=1)
                indices = indices.flatten()

                # 获取颜色
                vert_colors = sample_colors[indices].to(points.device)

                # 检查颜色是否有效
                if torch.isnan(vert_colors).any() or torch.isinf(vert_colors).any():
                    print("顶点颜色中检测到NaN或Inf，使用默认颜色")
                    vert_colors = torch.ones_like(vert_colors)  # 白色

                verts_colors.append(vert_colors)

            return verts_colors

        except Exception as e:
            print(f"颜色插值失败: {e}")

            # 创建默认颜色
            verts_colors = []
            for b in range(len(verts_list)):
                verts = verts_list[b]
                default_colors = torch.ones((verts.shape[0], 3), device=points.device)  # 白色
                verts_colors.append(default_colors)

            return verts_colors

    def convert_to_mesh(self, points, densities):
        """将点云和密度转换为网格 - 修复版本"""
        try:
            # 确保点云和密度是分离的，以避免梯度问题
            points_np = points.detach().cpu().numpy()
            densities_np = densities.detach().cpu().numpy()

            batch_size = points.shape[0]
            verts_list = []
            faces_list = []

            for b in range(batch_size):
                # 获取单个样本的点云和密度
                sample_points = points_np[b]
                sample_densities = densities_np[b]

                # 重塑为3D网格
                resolution = int(np.cbrt(sample_points.shape[0]))
                density_grid = sample_densities.reshape(resolution, resolution, resolution)

                # 使用Marching Cubes提取网格
                try:
                    import mcubes
                    vertices, triangles = mcubes.marching_cubes(density_grid, 0.5)

                    # 如果网格太小，抛出异常
                    if len(vertices) < 10 or len(triangles) < 10:
                        print("PyMCubes生成的网格太小，尝试使用scikit-image")
                        raise ValueError("Mesh too small")

                except Exception as e:
                    print(f"PyMCubes失败: {e}")
                    try:
                        from skimage import measure
                        vertices, faces, _, _ = measure.marching_cubes(density_grid, 0.5)
                        triangles = faces

                        # 如果网格太小，抛出异常
                        if len(vertices) < 10 or len(triangles) < 10:
                            raise ValueError("Mesh too small")

                    except Exception as e2:
                        print(f"scikit-image也失败了: {e2}，使用默认立方体")
                        # 创建一个简单的立方体作为后备
                        vertices = np.array([
                            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
                        ]) * 0.5
                        triangles = np.array([
                            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                            [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]
                        ])

                # 转换顶点坐标从体素索引到3D空间坐标
                voxel_size = 2.0 / resolution
                vertices = vertices * voxel_size - 1.0

                # 转换为PyTorch张量
                verts = torch.tensor(vertices, dtype=torch.float32, device=points.device)
                faces = torch.tensor(triangles, dtype=torch.int64, device=points.device)

                verts_list.append(verts)
                faces_list.append(faces)

            return verts_list, faces_list

        except Exception as e:
            print(f"网格转换失败: {e}")
            import traceback
            traceback.print_exc()

            # 创建一个简单的立方体作为后备
            batch_size = points.shape[0]
            verts_list = []
            faces_list = []

            for b in range(batch_size):
                verts = torch.tensor([
                    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
                ], dtype=torch.float32, device=points.device)

                faces = torch.tensor([
                    [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                    [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]
                ], dtype=torch.int64, device=points.device)

                verts_list.append(verts)
                faces_list.append(faces)

            return verts_list, faces_list

    def _create_default_meshes(self, batch_size, device):
        """创建默认网格列表"""
        verts_list = []
        faces_list = []

        for _ in range(batch_size):
            verts, faces = self._create_default_cube(device)
            verts_list.append(verts)
            faces_list.append(faces)

        return verts_list, faces_list

    def _simplify_mesh(self, verts, faces, target_faces=5000):
        """简化网格，减少面片数量"""
        try:
            import open3d as o3d

            # 创建Open3D网格
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts.cpu().numpy())
            mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())

            # 计算简化比例
            current_faces = len(faces)
            reduction = target_faces / current_faces if current_faces > target_faces else 1.0

            # 简化网格
            if reduction < 1.0:
                mesh = mesh.simplify_quadric_decimation(int(current_faces * reduction))

            # 转换回PyTorch张量
            simplified_verts = torch.tensor(np.asarray(mesh.vertices),
                                            dtype=torch.float32, device=verts.device)
            simplified_faces = torch.tensor(np.asarray(mesh.triangles),
                                            dtype=torch.long, device=faces.device)

            return simplified_verts, simplified_faces
        except Exception as e:
            print(f"网格简化失败: {e}")
            return verts, faces  # 返回原始网格

    def validate_mesh(self, mesh):
        """
        验证网格的有效性

        Args:
            mesh: PyTorch3D Meshes对象
        Returns:
            is_valid: 布尔值，表示网格是否有效
            message: 字符串，如果无效，包含错误信息
        """
        try:
            # 检查网格是否为空
            if mesh.isempty():
                return False, "网格为空"

            # 检查顶点数量
            if mesh.verts_padded().shape[1] < 3:
                return False, "顶点数量不足"

            # 检查面片数量
            if mesh.faces_padded().shape[1] < 1:
                return False, "面片数量不足"

            # 检查NaN值
            if torch.isnan(mesh.verts_padded()).any():
                return False, "顶点坐标包含NaN值"

            # 检查面片索引是否有效
            max_vert_idx = mesh.verts_padded().shape[1] - 1
            if (mesh.faces_padded() > max_vert_idx).any():
                return False, "面片索引超出顶点范围"

            # 检查网格是否可渲染（尝试渲染）
            try:
                _ = self._render_mesh(mesh)
            except Exception as e:
                return False, f"网格不可渲染: {e}"

            return True, "网格有效"

        except Exception as e:
            return False, f"网格验证过程出错: {e}"

    def validation_step(self, batch, batch_idx):
        """验证步骤，根据条件执行不同类型的验证"""
        # 基本的损失验证（每次都执行）
        val_loss_result = self._validate_with_loss(batch, batch_idx)

        # 条件性执行网格验证（每N个epoch或特定批次）
        do_mesh_validation = (self.current_epoch % 5 == 0) or (batch_idx == 0 and self.current_epoch == 0)

        if do_mesh_validation and batch_idx == 0:  # 仅对第一个批次执行网格验证
            mesh_val_result = self._validate_with_mesh(batch, batch_idx)
            # 合并结果
            val_loss_result.update(mesh_val_result)

        return val_loss_result

    def _validate_with_loss(self, batch, batch_idx):
        """基于损失的快速验证 - 使用训练模式但不更新参数"""
        # 关键修改：保持模型在训练模式
        self.train()  # 使用训练模式而不是eval模式

        # 确保不会更新参数
        with torch.no_grad():
            # 获取输入数据
            images = batch['image']
            target_points = batch['points']
            text_prompts = batch.get('text_prompts', None)

        # 重新启用梯度计算，但仅用于前向传播
        with torch.enable_grad():
            try:
                # 前向传播 - 完全匹配训练流程
                features_list = self.model.extract_features(images)
                fused_features, feature_weights, lighting_params = self.model.implicit_field.feature_pyramid(
                    features_list)

                # 创建点云
                batch_size = images.shape[0]
                points = self.model.implicit_field._create_grid_points(batch_size, images.device)
                points.requires_grad_(True)  # 确保点云需要梯度

                print(f"验证: 点云需要梯度: {points.requires_grad}")

                # 计算密度和特征
                densities, features = self.model.implicit_field.compute_density_and_features(points, fused_features)
                print(f"验证: 密度需要梯度: {densities.requires_grad}, grad_fn: {densities.grad_fn}")

                # 检查密度中的NaN
                if torch.isnan(densities).any():
                    print(f"验证: 密度中检测到NaN，尝试修复")
                    densities = torch.where(torch.isnan(densities),
                                            torch.ones_like(densities) * 1e-6,
                                            densities)

                # 计算颜色
                albedo = self.model.implicit_field.color_net(features)

                # 计算法线 - 使用与训练相同的方法
                normals = self.model.implicit_field.compute_normals(points, densities)
                print(f"验证: 法线需要梯度: {normals.requires_grad}, grad_fn: {normals.grad_fn}")

                # 应用光照
                colors = self.model.implicit_field.lighting_model.apply_lighting(
                    points, normals, albedo, lighting_params
                )

                # 提取CLIP特征
                clip_features = None
                if hasattr(self, 'clip_encoder'):
                    clip_features = self.clip_encoder(images)

                # 提取密度场特征
                density_features = self.model.implicit_field.extract_density_features(
                    points, densities, fused_features, clip_features
                )

                # 计算损失
                loss_dict = self.loss_fn({
                    'pred_points': points,
                    'pred_densities': densities,
                    'pred_colors': colors,
                    'pred_normals': normals,
                    'density_features': density_features,
                    'original_images': images,
                }, {
                    'target_points': target_points,
                    'text_prompts': text_prompts,
                }, model=self.model)

                # 记录各项损失
                val_loss = loss_dict['total']
                self.log("val_loss", val_loss.item(), prog_bar=True)

                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        self.log(f"val_{key}", value.item())
                    else:
                        self.log(f"val_{key}", value)

                return {"val_loss": val_loss.detach()}

            except Exception as e:
                print(f"验证: 出错: {e}")
                import traceback
                traceback.print_exc()
                return {"val_loss": torch.tensor(1.0, device=self.device)}

    def _validate_with_mesh(self, batch, batch_idx):
        """基于网格重建的完整验证 - 修复版本"""
        # 保存当前模式
        was_training = self.training

        # 临时设置为训练模式
        self.train()  # 使用训练模式而不是eval模式

        images = batch['image']
        target_points = batch['points']
        batch_size = images.shape[0]

        try:
            # 启用梯度计算
            with torch.enable_grad():
                # 生成网格
                meshes = self.generate_mesh(images, is_training=False)

                # 验证网格有效性
                is_valid = len(meshes) > 0 and meshes.verts_padded().shape[1] > 0
                message = "有效网格" if is_valid else "无效网格"

                self.log("val_mesh_valid", float(is_valid), prog_bar=True, batch_size=batch_size)

                if is_valid:
                    # 计算Chamfer距离
                    metrics = Metrics(device=self.device)
                    pred_points = meshes.verts_padded()
                    cd = metrics.compute_chamfer_distance(pred_points, target_points)
                    self.log("val_mesh_chamfer", cd, prog_bar=True, batch_size=batch_size)

                    # 渲染网格
                    rendered_images = self._render_mesh_safe(meshes)

                    # 保存可视化结果（仅对第一个批次）
                    if batch_idx == 0:
                        self._save_visualization(rendered_images, self.current_epoch)

                    return {"val_mesh_chamfer": cd}
                else:
                    print(f"网格验证失败: {message}")
                    return {"val_mesh_valid": 0.0}
        except Exception as e:
            print(f"网格评估过程出错: {e}")
            import traceback
            traceback.print_exc()
            return {"val_mesh_valid": 0.0}
        finally:
            # 恢复原始模式
            if not was_training:
                self.eval()

    def _create_default_cube(self, device):
        """创建默认立方体网格"""
        # 定义立方体的8个顶点
        verts = torch.tensor([
            [-0.5, -0.5, -0.5],  # 0
            [0.5, -0.5, -0.5],  # 1
            [-0.5, 0.5, -0.5],  # 2
            [0.5, 0.5, -0.5],  # 3
            [-0.5, -0.5, 0.5],  # 4
            [0.5, -0.5, 0.5],  # 5
            [-0.5, 0.5, 0.5],  # 6
            [0.5, 0.5, 0.5]  # 7
        ], dtype=torch.float32, device=device)

        # 定义立方体的12个三角形面（6个正方形面，每个分成2个三角形）
        faces = torch.tensor([
            [0, 1, 2],  # 底面1
            [1, 3, 2],  # 底面2
            [4, 6, 5],  # 顶面1
            [5, 6, 7],  # 顶面2
            [0, 4, 1],  # 前面1
            [1, 4, 5],  # 前面2
            [2, 3, 6],  # 后面1
            [3, 7, 6],  # 后面2
            [0, 2, 4],  # 左面1
            [2, 6, 4],  # 左面2
            [1, 5, 3],  # 右面1
            [3, 5, 7]  # 右面2
        ], dtype=torch.long, device=device)

        return verts, faces

    def _sync_device(self):
        """确保所有组件使用相同的设备"""
        current_device = self.device
        print(f"同步所有组件到设备: {current_device}")

        # 同步 CLIP 编码器
        if hasattr(self, 'clip_encoder'):
            self.clip_encoder = self.clip_encoder.to(current_device)
            print(f"CLIP 编码器已移至设备: {current_device}")

        # 同步渲染器
        if hasattr(self, 'renderer'):
            self.renderer = self.setup_renderer()
            print(f"渲染器已重新初始化在设备: {current_device}")

        # 同步损失函数
        if hasattr(self.loss_fn, 'to'):
            self.loss_fn = self.loss_fn.to(current_device)
            print(f"损失函数已移至设备: {current_device}")
        elif hasattr(self.loss_fn, 'clip_loss') and hasattr(self.loss_fn.clip_loss, 'clip_model'):
            self.loss_fn.clip_loss.clip_model = self.loss_fn.clip_loss.clip_model.to(current_device)
            print(f"CLIP 损失模型已移至设备: {current_device}")


# 主函数
def main():
    # 配置参数
    transform = DataAugmentation()
    cfg = SimpleNamespace(**{
        'data_root': None,
        'categories': ['airplane'],
        'batch_size': 8,
        'num_workers': 4,
        'lr': 1e-4,
        'transform': None,
        'max_epochs': 100,
        'lambda_depth': 0.5,
        'lambda_clip': 0.3,
        'lambda_edge': 0.2,
        'lambda_reg': 0.01,
        'lambda_lap': 0.1,
        'beta_grad': 0.05,
        'temperature': 0.07,
        'clip_model': 'ViT-B/32',
        'render_size': 128,  # 添加渲染大小参数
        'samples_per_category': 200,
        'mesh_eval_frequency': 5,
        'early_stop_patience': 10,
        'gpu': 0
    })

    # 创建训练器
    trainer = CLIPNeRFTrainer(cfg)

    # # 创建logger
    # logger = WandbLogger(project='clip-nerf')
    # 创建 WandB Logger 并设置模式
    logger = WandbLogger(
        project='clip-nerf',
        name=f'run_{cfg.categories[0]}',  # 可以根据类别设置运行名称
        mode='offline',  # 可以是 'online', 'offline', 'disabled'
        config=vars(cfg)  # 直接传入配置
    )


    # 创建训练器

    # 创建检查点回调
    # 创建检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',  # 确保这与validation_step中记录的指标一致
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_epochs=1
    )

    # 创建早停回调
    early_stop_callback = EarlyStopping(
        monitor='val_loss',  # 与checkpoint_callback监控相同的指标
        patience=cfg.early_stop_patience,
        mode='min',
        verbose=True,
    )
    pl_trainer = pl.Trainer(
        accelerator='gpu',
        devices=[cfg.gpu],
        precision=32,
        gradient_clip_val=1.0,  # 在训练器级别设置梯度裁剪
        max_epochs=cfg.max_epochs,
        # gpus=2,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
    )


    # pl_trainer = pl.Trainer(
    #     max_epochs=cfg.max_epochs,
    #     gpus=1,
    #     logger=logger,
    #     precision=16,  # 混合精度训练
    #     gradient_clip_val=1.0
    # )

    # 开始训练
    pl_trainer.fit(trainer)


if __name__ == '__main__':
    main()
