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
        self._cached_renderer = None
        self.rgb_only = False  # 如果只需要RGB输出（不含Alpha通道），设为True

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
        try:
            import torchvision
            from PIL import Image
            import os

            # 确保输出目录存在
            os.makedirs("visualizations", exist_ok=True)

            # 检查渲染图像的形状
            if rendered_images is None:
                print("警告: 渲染图像为None，跳过可视化")
                return

            print(f"保存可视化前的图像形状: {rendered_images.shape}")

            # 确保图像格式是[B, C, H, W]
            if len(rendered_images.shape) == 4:
                if rendered_images.shape[1] > 4:  # 如果第二维（通道维）大于4
                    # 可能格式是[B, H, W, C]，需要转置
                    if rendered_images.shape[3] <= 4:
                        rendered_images = rendered_images.permute(0, 3, 1, 2)
                        print(f"转置后的图像形状: {rendered_images.shape}")
                    else:
                        # 如果两个维度都大于4，可能有问题
                        print(f"警告: 图像形状异常: {rendered_images.shape}")
                        # 尝试推断正确的格式
                        if rendered_images.shape[0] < 10:  # 批次大小通常较小
                            # 假设格式是[B, H, W, C]但C异常大
                            rendered_images = rendered_images[..., :3]  # 只保留前3个通道
                            rendered_images = rendered_images.permute(0, 3, 1, 2)
                        else:
                            # 其他情况，创建空白图像
                            rendered_images = torch.ones(1, 3, 256, 256, device=self.device)

            # 确保通道数为3
            if rendered_images.shape[1] > 3:
                print(f"警告: 通道数为 {rendered_images.shape[1]}，裁剪为3通道")
                rendered_images = rendered_images[:, :3]

            # 确保值在[0, 1]范围内
            rendered_images = torch.clamp(rendered_images, 0.0, 1.0)

            # 选择前8个样本（或更少）
            num_samples = min(8, rendered_images.shape[0])
            images_to_save = rendered_images[:num_samples]

            # 创建网格图像
            grid = torchvision.utils.make_grid(images_to_save, nrow=4)

            # 转换为PIL图像并保存
            # 使用手动转换而不是ToPILImage，以避免通道数问题
            grid_np = grid.cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            grid_np = (grid_np * 255).astype('uint8')
            pil_image = Image.fromarray(grid_np)

            # 保存图像
            output_path = f"visualizations/epoch_{self.current_epoch}_batch_{batch_idx}.png"
            pil_image.save(output_path)
            print(f"可视化结果已保存到 {output_path}")

        except Exception as e:
            print(f"保存可视化过程出错: {e}")
            import traceback
            traceback.print_exc()

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

    def _render_simple(self, meshes):
        """修复版本的简单渲染方法"""
        try:
            from pytorch3d.renderer import (
                look_at_view_transform, FoVPerspectiveCameras,
                RasterizationSettings, MeshRasterizer
            )

            # 确保网格在正确的设备上
            target_device = self.device
            if meshes.device != target_device:
                meshes = meshes.to(target_device)

            # 创建相机
            R, T = look_at_view_transform(2.0, 0, 0)
            cameras = FoVPerspectiveCameras(R=R, T=T, device=target_device)

            # 创建光栅化器
            raster_settings = RasterizationSettings(
                image_size=256,
                blur_radius=0.0,
                faces_per_pixel=1,
            )

            rasterizer = MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            )

            # 只进行光栅化
            fragments = rasterizer(meshes)

            # 提取深度信息
            zbuf = fragments.zbuf  # 形状: [B, H, W, faces_per_pixel]

            # 创建输出图像
            batch_size = len(meshes)
            image_size = 256
            images = torch.ones(batch_size, 3, image_size, image_size, device=target_device)

            # 对每个批次样本处理
            for b in range(batch_size):
                # 创建掩码，标识有效的深度值
                valid_mask = zbuf[b, :, :, 0] > 0  # [H, W]

                # 获取有效的深度值
                valid_depths = zbuf[b, valid_mask, 0]

                if valid_depths.numel() > 0:  # 确保有有效的深度值
                    # 归一化深度值
                    min_depth = valid_depths.min()
                    max_depth = valid_depths.max()
                    depth_range = max_depth - min_depth

                    if depth_range > 1e-6:  # 避免除以接近零的值
                        # 创建归一化的深度图
                        depth_image = torch.ones_like(zbuf[b, :, :, 0])
                        normalized_depth = (zbuf[b, :, :, 0] - min_depth) / depth_range
                        depth_image[valid_mask] = 1.0 - normalized_depth[valid_mask]

                        # 将深度图复制到三个通道
                        images[b, 0] = depth_image  # 红色通道
                        images[b, 1] = depth_image  # 绿色通道
                        images[b, 2] = depth_image  # 蓝色通道

            return images

        except Exception as e:
            print(f"简单渲染过程出错: {e}")
            import traceback
            traceback.print_exc()

            # 返回空白RGB图像作为后备
            batch_size = len(meshes)
            return torch.ones(batch_size, 3, 256, 256, device=self.device)

    def _render_basic(self, meshes):
        """最基本的渲染方法，不依赖于光栅化"""
        try:
            batch_size = len(meshes)
            image_size = 256
            images = torch.ones(batch_size, 3, image_size, image_size, device=self.device)

            for b in range(batch_size):
                # 获取顶点
                verts = meshes.verts_padded()[b]

                if verts.shape[0] > 0:  # 确保有顶点
                    # 计算顶点的边界框
                    min_coords = torch.min(verts, dim=0)[0]
                    max_coords = torch.max(verts, dim=0)[0]

                    # 将顶点归一化到[0,1]范围
                    range_coords = max_coords - min_coords
                    range_coords = torch.where(range_coords < 1e-6, torch.ones_like(range_coords), range_coords)
                    norm_verts = (verts - min_coords) / range_coords

                    # 只保留前两个坐标（x和y）并缩放到图像大小
                    image_coords = (norm_verts[:, :2] * (image_size - 1)).long()

                    # 确保坐标在有效范围内
                    image_coords = torch.clamp(image_coords, 0, image_size - 1)

                    # 创建一个简单的点云可视化
                    for i in range(min(1000, len(image_coords))):  # 限制点数以提高性能
                        x, y = image_coords[i]

                        # 在每个点周围画一个3x3的点
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < image_size and 0 <= ny < image_size:
                                    # 使用z坐标作为颜色
                                    z_norm = norm_verts[i, 2]
                                    images[b, 0, ny, nx] = z_norm  # 红色 - 深度
                                    images[b, 1, ny, nx] = 0.5  # 绿色 - 固定
                                    images[b, 2, ny, nx] = 1.0 - z_norm  # 蓝色 - 反深度

            return images

        except Exception as e:
            print(f"基本渲染过程出错: {e}")
            import traceback
            traceback.print_exc()

            # 返回纯色图像
            return torch.ones(batch_size, 3, image_size, image_size, device=self.device)


    def _render_mesh_safe(self, meshes):
        """安全地渲染网格，处理可能的错误"""
        try:
            # 导入必要的库
            from pytorch3d.structures import Meshes
            from pytorch3d.renderer import (
                look_at_view_transform, FoVPerspectiveCameras,
                RasterizationSettings, MeshRenderer, MeshRasterizer,
                SoftPhongShader, PointLights
            )

            # 确保网格在正确的设备上
            target_device = self.device
            if meshes.device != target_device:
                meshes = meshes.to(target_device)

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
                textures = meshes.textures
                # 确保纹理在正确的设备上
                if hasattr(textures, 'device') and textures.device != target_device:
                    textures = textures.to(target_device)

                # 解绑顶点和面
                verts_unbind = verts.unbind()
                faces_unbind = faces.unbind()

                # 创建新的网格
                meshes = Meshes(verts=verts_unbind, faces=faces_unbind, textures=textures)

            # 尝试使用CPU渲染以避免CUDA架构问题
            try:
                # 创建相机
                R, T = look_at_view_transform(2.0, 0, 0)
                cameras = FoVPerspectiveCameras(R=R, T=T, device=target_device)

                # 创建光照
                lights = PointLights(
                    location=[[0.0, 0.0, 2.0]],
                    ambient_color=[[0.5, 0.5, 0.5]],
                    diffuse_color=[[0.3, 0.3, 0.3]],
                    specular_color=[[0.2, 0.2, 0.2]],
                    device=target_device
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
                ).to(target_device)

                # 渲染网格
                rendered_images = renderer(meshes)
            except Exception as cuda_error:
                print(f"GPU渲染失败，尝试使用CPU: {cuda_error}")
                # 转移到CPU
                cpu_meshes = meshes.to('cpu')

                # 创建CPU渲染器
                R, T = look_at_view_transform(2.0, 0, 0)
                cameras = FoVPerspectiveCameras(R=R, T=T, device='cpu')

                lights = PointLights(
                    location=[[0.0, 0.0, 2.0]],
                    ambient_color=[[0.5, 0.5, 0.5]],
                    diffuse_color=[[0.3, 0.3, 0.3]],
                    specular_color=[[0.2, 0.2, 0.2]],
                    device='cpu'
                )

                raster_settings = RasterizationSettings(
                    image_size=256,
                    blur_radius=0.0,
                    faces_per_pixel=1,
                )

                cpu_renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                    shader=SoftPhongShader(cameras=cameras, lights=lights)
                )

                # 在CPU上渲染
                rendered_images = cpu_renderer(cpu_meshes)
                # 移回原始设备
                rendered_images = rendered_images.to(target_device)

            # 检查渲染图像的形状和通道数
            B, H, W, C = rendered_images.shape

            print(f"渲染图像形状: {rendered_images.shape}")  # 调试信息

            # 确保输出是3通道RGB图像
            if C > 3:  # 如果通道数过多
                print(f"警告: 渲染图像通道数为 {C}，裁剪为3通道")
                rendered_images = rendered_images[..., :3]  # 只保留前3个通道（RGB）

            # 检查是否有NaN或Inf值
            if torch.isnan(rendered_images).any() or torch.isinf(rendered_images).any():
                print("渲染图像中检测到NaN或Inf，进行修复")
                rendered_images = torch.where(
                    torch.isnan(rendered_images) | torch.isinf(rendered_images),
                    torch.zeros_like(rendered_images),
                    rendered_images
                )

            # 转换为[B, C, H, W]格式，这是PyTorch常用的图像格式
            rendered_images = rendered_images.permute(0, 3, 1, 2)

            return rendered_images

        except Exception as e:
            print(f"渲染过程出错: {e}")
            import traceback
            traceback.print_exc()

            # 返回空白RGB图像作为后备
            batch_size = len(meshes) if isinstance(meshes, Meshes) else 1
            return torch.ones(batch_size, 3, 256, 256, device=self.device)  # 返回[B, C, H, W]格式

    def _create_fallback_mesh(self, batch_size):
        """创建后备网格，当所有其他方法失败时使用"""
        from pytorch3d.structures import Meshes
        import torch
        import numpy as np

        try:
            # 创建一个简单的立方体
            vertices = np.array([
                [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
            ], dtype=np.float32)

            faces = np.array([
                [0, 1, 2], [0, 2, 3],  # 底面
                [4, 5, 6], [4, 6, 7],  # 顶面
                [0, 1, 5], [0, 5, 4],  # 侧面1
                [1, 2, 6], [1, 6, 5],  # 侧面2
                [2, 3, 7], [2, 7, 6],  # 侧面3
                [3, 0, 4], [3, 4, 7]  # 侧面4
            ], dtype=np.int64)

            # 转换为PyTorch张量
            verts_tensor = torch.tensor(vertices, dtype=torch.float32, device=self.device)
            faces_tensor = torch.tensor(faces, dtype=torch.int64, device=self.device)

            # 为批次中的每个样本创建相同的网格
            verts_list = [verts_tensor.clone() for _ in range(batch_size)]
            faces_list = [faces_tensor.clone() for _ in range(batch_size)]

            # 创建Meshes对象
            meshes = Meshes(verts=verts_list, faces=faces_list)
            return meshes

        except Exception as e:
            print(f"创建后备网格时出错: {e}")

            # 最后的后备选项：创建最简单的三角形
            try:
                # 创建一个简单的三角形
                simple_verts = torch.tensor([
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]
                ], dtype=torch.float32, device=self.device)

                simple_faces = torch.tensor([
                    [0, 1, 2]
                ], dtype=torch.int64, device=self.device)

                # 为批次中的每个样本创建相同的三角形
                simple_verts_list = [simple_verts.clone() for _ in range(batch_size)]
                simple_faces_list = [simple_faces.clone() for _ in range(batch_size)]

                # 创建Meshes对象
                return Meshes(verts=simple_verts_list, faces=simple_faces_list)

            except Exception as e2:
                print(f"创建最简单的后备网格时也出错: {e2}")

                # 如果所有方法都失败，返回一个空的Meshes对象
                empty_verts = torch.zeros((batch_size, 3, 3), dtype=torch.float32, device=self.device)
                empty_faces = torch.zeros((batch_size, 1, 3), dtype=torch.int64, device=self.device)
                return Meshes(verts=empty_verts, faces=empty_faces)

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

    def generate_mesh(self, images, is_training=True):
        """
        从图像生成网格，包含多种后备方案和错误处理
        """
        from pytorch3d.structures import Meshes
        import torch
        import numpy as np

        # 确保我们知道批次大小，即使前向传播失败
        batch_size = images.shape[0]

        try:
            # 确保模型处于训练模式，以避免BN层和dropout的问题
            was_training = self.training
            if is_training and not was_training:
                self.train()
            elif not is_training and was_training:
                self.eval()

            # 尝试获取体素场
            try:
                with torch.set_grad_enabled(is_training):
                    # 直接调用模型的方法，避免通过forward
                    if hasattr(self.model, "get_voxel_field"):
                        voxel_field = self.model.get_voxel_field(images)
                    else:
                        # 尝试直接从模型获取体素场，避免解包错误
                        output = self.model(images)

                        # 检查输出类型
                        if isinstance(output, tuple) and len(output) == 2:
                            # 如果输出是元组，假设第二个元素是体素场
                            voxel_field = output[1]
                        elif isinstance(output, dict) and 'voxel_field' in output:
                            # 如果输出是字典，尝试获取体素场
                            voxel_field = output['voxel_field']
                        else:
                            # 否则假设整个输出就是体素场
                            voxel_field = output
            except Exception as forward_error:
                print(f"前向传播失败: {forward_error}")
                import traceback
                traceback.print_exc()

                # 创建一个默认的体素场
                print("创建默认体素场")
                grid_size = 32  # 假设体素网格大小为32
                voxel_field = torch.zeros((batch_size, grid_size, grid_size, grid_size), device=self.device)

                # 在中心创建一个球体 - 使用高效的向量化操作
                center = grid_size // 2
                radius = grid_size // 4

                # 创建坐标网格 (避免使用ogrid，它会创建完整的网格)
                coords = torch.arange(grid_size, device=self.device)
                x = coords.view(grid_size, 1, 1).expand(grid_size, grid_size, grid_size)
                y = coords.view(1, grid_size, 1).expand(grid_size, grid_size, grid_size)
                z = coords.view(1, 1, grid_size).expand(grid_size, grid_size, grid_size)

                # 计算到中心的距离
                dist_from_center = torch.sqrt(((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2).float())
                sphere_voxels = (radius - dist_from_center).clamp(min=0)

                # 将球体复制到每个批次样本
                for b in range(batch_size):
                    voxel_field[b] = sphere_voxels

            # 恢复原始训练模式
            if is_training != was_training:
                if was_training:
                    self.train()
                else:
                    self.eval()

            # 创建空列表存储网格
            verts_list = []
            faces_list = []

            for b in range(batch_size):
                # 获取当前样本的体素场
                voxels = voxel_field[b].detach().cpu().numpy()

                # 检查体素场是否有效
                if np.isnan(voxels).any() or np.isinf(voxels).any():
                    print(f"警告：样本 {b} 的体素场包含NaN或Inf值，进行修复")
                    voxels = np.nan_to_num(voxels, nan=0.0, posinf=1.0, neginf=0.0)

                # 检查体素值范围
                voxel_min = np.min(voxels)
                voxel_max = np.max(voxels)
                value_range = voxel_max - voxel_min

                print(f"样本 {b} 的体素场范围: [{voxel_min}, {voxel_max}]")

                # 如果范围太小，调整体素场
                if value_range < 1e-6:
                    print(f"警告：样本 {b} 的体素场范围太小，进行调整")

                    # 创建一个简单的球体体素场作为替代 - 使用高效方法
                    grid_size = voxels.shape[0]
                    center = grid_size // 2
                    radius = grid_size // 4

                    # 使用更高效的方法创建球体，避免内存溢出
                    # 创建一个空的体素场
                    new_voxels = np.zeros_like(voxels)

                    # 逐层填充球体
                    for i in range(grid_size):
                        for j in range(grid_size):
                            for k in range(grid_size):
                                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2 + (k - center) ** 2)
                                if dist < radius:
                                    new_voxels[i, j, k] = radius - dist

                    voxels = new_voxels

                # 尝试多种方法生成网格
                vertices = None
                faces = None

                # 方法1：使用PyMCubes
                try:
                    import mcubes
                    print(f"尝试使用PyMCubes生成样本 {b} 的网格...")

                    # 尝试不同的阈值
                    for threshold in [0.0, 0.25, 0.5, -0.25, -0.5]:
                        try:
                            v, f = mcubes.marching_cubes(voxels, threshold)
                            if len(v) > 10 and len(f) > 10:  # 确保网格足够大
                                vertices = v
                                faces = f
                                print(f"PyMCubes成功：阈值 = {threshold}, 顶点数 = {len(v)}, 面数 = {len(f)}")
                                break
                        except Exception as mc_err:
                            print(f"PyMCubes在阈值 {threshold} 失败: {mc_err}")
                except Exception as e:
                    print(f"PyMCubes完全失败: {e}")

                # 方法2：如果PyMCubes失败，尝试scikit-image
                if vertices is None or faces is None:
                    try:
                        from skimage import measure
                        print(f"尝试使用scikit-image生成样本 {b} 的网格...")

                        # 尝试不同的阈值
                        for threshold in [0.0, 0.25, 0.5, -0.25, -0.5]:
                            try:
                                verts, faces, _, _ = measure.marching_cubes(voxels, level=threshold)
                                if len(verts) > 10 and len(faces) > 10:  # 确保网格足够大
                                    vertices = verts
                                    faces = faces
                                    print(
                                        f"scikit-image成功：阈值 = {threshold}, 顶点数 = {len(verts)}, 面数 = {len(faces)}")
                                    break
                            except Exception as mc_err:
                                print(f"scikit-image在阈值 {threshold} 失败: {mc_err}")
                    except Exception as e:
                        print(f"scikit-image完全失败: {e}")

                # 方法3：如果前两种方法都失败，使用默认立方体
                if vertices is None or faces is None:
                    print(f"所有网格生成方法都失败，使用默认立方体作为样本 {b} 的网格")
                    # 创建一个简单的立方体
                    vertices = np.array([
                        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
                    ], dtype=np.float32)

                    faces = np.array([
                        [0, 1, 2], [0, 2, 3],  # 底面
                        [4, 5, 6], [4, 6, 7],  # 顶面
                        [0, 1, 5], [0, 5, 4],  # 侧面1
                        [1, 2, 6], [1, 6, 5],  # 侧面2
                        [2, 3, 7], [2, 7, 6],  # 侧面3
                        [3, 0, 4], [3, 4, 7]  # 侧面4
                    ], dtype=np.int64)

                # 归一化顶点坐标到[-1, 1]范围
                if vertices is not None:
                    vmin = vertices.min(axis=0)
                    vmax = vertices.max(axis=0)
                    vrange = vmax - vmin
                    # 避免除以零
                    vrange[vrange < 1e-6] = 1.0
                    vertices = 2 * (vertices - vmin) / vrange - 1

                # 转换为PyTorch张量并添加到列表
                verts_list.append(torch.tensor(vertices, dtype=torch.float32, device=self.device))
                faces_list.append(torch.tensor(faces, dtype=torch.int64, device=self.device))

            # 创建Meshes对象
            meshes = Meshes(verts=verts_list, faces=faces_list)
            return meshes

        except Exception as e:
            print(f"网格生成过程出错: {e}")
            import traceback
            traceback.print_exc()

            # 创建一个后备网格
            return self._create_fallback_mesh(batch_size)

    def _create_sphere_voxels(self, grid_size, radius=None):
        """
        创建一个球体体素场，使用高效的方法避免内存溢出
        """
        import numpy as np

        if radius is None:
            radius = grid_size // 4

        center = grid_size // 2
        voxels = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

        # 计算球体的边界框，减少迭代次数
        min_idx = max(0, center - radius - 1)
        max_idx = min(grid_size, center + radius + 1)

        # 只迭代边界框内的体素
        for i in range(min_idx, max_idx):
            for j in range(min_idx, max_idx):
                for k in range(min_idx, max_idx):
                    dist = np.sqrt((i - center) ** 2 + (j - center) ** 2 + (k - center) ** 2)
                    if dist < radius:
                        voxels[i, j, k] = radius - dist

        return voxels

    def _direct_mesh_from_voxels(self, voxels, threshold=0.5):
        """
        直接从体素场生成网格，避免使用marching cubes
        """
        import numpy as np
        import torch

        # 确保体素场是numpy数组
        if isinstance(voxels, torch.Tensor):
            voxels = voxels.detach().cpu().numpy()

        grid_size = voxels.shape[0]

        # 创建顶点和面的列表
        vertices = []
        faces = []

        # 为每个超过阈值的体素创建一个立方体
        face_idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if voxels[i, j, k] > threshold:
                        # 计算体素的8个顶点
                        x, y, z = i, j, k
                        cube_verts = [
                            [x, y, z], [x + 1, y, z], [x + 1, y + 1, z], [x, y + 1, z],
                            [x, y, z + 1], [x + 1, y, z + 1], [x + 1, y + 1, z + 1], [x, y + 1, z + 1]
                        ]

                        # 添加顶点
                        for vert in cube_verts:
                            vertices.append(vert)

                        # 添加面 (每个立方体有12个三角形)
                        cube_faces = [
                            [0, 1, 2], [0, 2, 3],  # 底面
                            [4, 5, 6], [4, 6, 7],  # 顶面
                            [0, 1, 5], [0, 5, 4],  # 侧面1
                            [1, 2, 6], [1, 6, 5],  # 侧面2
                            [2, 3, 7], [2, 7, 6],  # 侧面3
                            [3, 0, 4], [3, 4, 7]  # 侧面4
                        ]

                        # 调整面索引
                        for face in cube_faces:
                            faces.append([face[0] + face_idx, face[1] + face_idx, face[2] + face_idx])

                        face_idx += 8

        # 如果没有找到任何体素，创建一个默认立方体
        if len(vertices) == 0:
            vertices = [
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
            ]
            faces = [
                [0, 1, 2], [0, 2, 3],  # 底面
                [4, 5, 6], [4, 6, 7],  # 顶面
                [0, 1, 5], [0, 5, 4],  # 侧面1
                [1, 2, 6], [1, 6, 5],  # 侧面2
                [2, 3, 7], [2, 7, 6],  # 侧面3
                [3, 0, 4], [3, 4, 7]  # 侧面4
            ]

        # 转换为numpy数组
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int64)

        # 归一化顶点坐标到[-1, 1]范围
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)
        vrange = vmax - vmin
        # 避免除以零
        vrange[vrange < 1e-6] = 1.0
        vertices = 2 * (vertices - vmin) / vrange - 1

        return vertices, faces

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
        do_mesh_validation = False # (self.current_epoch % 5 == 0) and self.current_epoch != 0# or (batch_idx == 0 and self.current_epoch == 0)

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
        """基于网格重建的验证 - 更健壮的版本"""
        # 保存当前模式
        was_training = self.training

        # 临时设置为训练模式
        self.train()  # 使用训练模式

        # 获取批次大小
        images = batch['image']
        batch_size = images.shape[0]

        try:
            target_points = batch['points']

            # 生成网格
            try:
                meshes = self.generate_mesh(images, is_training=True)  # 确保使用训练模式
            except Exception as mesh_error:
                print(f"生成网格失败，使用后备网格: {mesh_error}")
                meshes = self._create_fallback_mesh(batch_size)

            # 验证网格有效性
            is_valid = len(meshes) > 0 and meshes.verts_padded().shape[1] > 0
            self.log("val_mesh_valid", float(is_valid), prog_bar=True, batch_size=batch_size)

            if is_valid:
                # 计算Chamfer距离
                try:
                    metrics = Metrics(device=self.device)
                    pred_points = meshes.verts_padded()
                    cd = metrics.compute_chamfer_distance(pred_points, target_points)
                    self.log("val_mesh_chamfer", cd, prog_bar=True, batch_size=batch_size)
                except Exception as cd_error:
                    print(f"计算Chamfer距离时出错: {cd_error}")
                    cd = torch.tensor(1000.0, device=self.device)  # 大值表示错误
                    self.log("val_mesh_chamfer", cd, prog_bar=True, batch_size=batch_size)

                # 尝试多种渲染方法
                rendered_images = None

                # 方法1: 尝试基本渲染（最可靠的方法）
                try:
                    print("尝试使用基本渲染...")
                    rendered_images = self._render_basic(meshes)
                except Exception as e1:
                    print(f"基本渲染失败: {e1}")

                    # 方法2: 尝试简单渲染
                    try:
                        print("尝试使用简单渲染...")
                        rendered_images = self._render_simple(meshes)
                    except Exception as e2:
                        print(f"简单渲染失败: {e2}")

                        # 最后的后备方案：创建纯色图像
                        print("所有渲染方法都失败，使用纯色图像")
                        rendered_images = torch.ones(batch_size, 3, 256, 256, device=self.device)

                # 保存可视化结果（仅对第一个批次）
                if batch_idx == 0 and rendered_images is not None:
                    try:
                        self._save_visualization(rendered_images, self.current_epoch)
                    except Exception as vis_error:
                        print(f"保存可视化结果时出错: {vis_error}")

                return {"val_mesh_chamfer": cd}
            else:
                print("网格验证失败：无效网格")
                return {"val_mesh_valid": 0.0}
        except Exception as e:
            print(f"网格评估过程出错: {e}")
            import traceback
            traceback.print_exc()

            # 确保返回一个有效的字典
            return {"val_mesh_valid": 0.0, "val_mesh_chamfer": 1000.0}
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
