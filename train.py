# train.py
from types import SimpleNamespace

import torch
import torch.optim as optim
import wandb
from pytorch3d.structures import Meshes
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, default_collate
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from models.base_model import SingleViewReconstructor
from models.clip_encoder import CLIPEncoder
from losses.combined_loss import CombinedLoss, chamfer_distance_loss
from data.shapenet import ShapeNetDataset
from utils.aug import DataAugmentation


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
        """在训练开始时设置渲染器"""
        self.renderer = self.setup_renderer()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.train()  # 确保模型在训练模式
        images = batch['image']
        target_points = batch['points']

        # 前向传播
        pred_points, pred_densities = self.generate_mesh(images, is_training=True)

        # 检查输出是否有 NaN
        if torch.isnan(pred_points).any() or torch.isnan(pred_densities).any():
            print(f"Warning: NaN detected in model outputs at batch {batch_idx}")
            # 创建一个与计算图相连的损失
            return {"loss": torch.sum(pred_points * 0) + torch.tensor(1e6, device=self.device, requires_grad=True)}

        # 计算损失
        loss_dict = self.loss_fn({
            'pred_points': pred_points,
            'pred_densities': pred_densities,
        }, {
            'target_points': target_points,
        })

        # 确保损失是标量且需要梯度
        if isinstance(loss_dict['total'], torch.Tensor):
            loss = loss_dict['total']
        else:
            loss = torch.tensor(loss_dict['total'], device=self.device, requires_grad=True)

        # 返回包含 'loss' 键的字典
        return {"loss": loss}

    def training_step_(self, batch, batch_idx):
        images = batch['image']
        target_points = batch['points']
        try:
            # 前向传播
            pred_points, pred_densities = self.generate_mesh(images, is_training=True)

            # 计算损失
            loss_dict = self.loss_fn({
                'pred_points': pred_points,
                'pred_densities': pred_densities,
            }, {
                'target_points': target_points,
            })

            # 检查损失值
            if torch.isnan(loss_dict['total']):
                print(f"Warning: NaN loss detected in batch {batch_idx}")
                return None

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            return loss_dict

        except ValueError as e:
            print(f"Error in batch {batch_idx}: {e}")
            return None

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
                "monitor": "train_loss",
                "frequency": 1
            }
        }

    def train_dataloader(self):
        # 创建ShapeNet数据集
        dataset = ShapeNetDataset(
            root_dir=self.cfg.data_root,
            split='train',
            categories=self.cfg.categories,
            transform=self.cfg.transform
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

    def generate_mesh(self, images, is_training=None):
        """
        从图像生成3D表示
        Args:
            images: 输入图像批次 [B, 3, H, W]
            is_training: 是否处于训练模式
        """
        if is_training is None:
            is_training = self.training

        # 获取模型输出
        points, densities = self.model(images)

        if is_training:
            # 训练时直接返回点云和密度
            return points, densities
        else:
            # 验证/测试时生成完整网格
            with torch.no_grad():
                verts, faces = self.convert_to_mesh(points, densities)
                return Meshes(verts=[v for v in verts], faces=[f for f in faces])

    def convert_to_mesh(self, points, densities):
        """
        将点云和密度转换为网格
        Args:
            points: 点云坐标 [B, N, 3]
            densities: 点云密度 [B, N, 1]
        Returns:
            verts: 顶点坐标 [B, V, 3]
            faces: 面片索引 [B, F, 3]
        """
        # TODO: 实现从点云到网格的转换
        # 可以使用 Marching Cubes 算法或其他网格重建方法
        # 这里需要根据具体的网格重建方法来实现
        # 临时返回一个简单的立方体网格作为示例
        batch_size = points.shape[0]
        device = points.device

        # 创建一个简单的立方体网格
        verts = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ], device=device).float()

        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],  # 前
            [4, 5, 6], [4, 6, 7],  # 后
            [0, 1, 5], [0, 5, 4],  # 下
            [2, 3, 7], [2, 7, 6],  # 上
            [0, 3, 7], [0, 7, 4],  # 左
            [1, 2, 6], [1, 6, 5],  # 右
        ], device=device).long()

        # 扩展到批次大小
        verts = verts.unsqueeze(0).expand(batch_size, -1, -1)
        faces = faces.unsqueeze(0).expand(batch_size, -1, -1)

        return verts, faces


# 主函数
def main():
    # 配置参数
    transform = DataAugmentation()
    cfg = SimpleNamespace(**{
        'data_root': None,
        'categories': ['airplane'],
        'batch_size': 4,
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
        'render_size': 128  # 添加渲染大小参数
    })

    # 创建训练器
    trainer = CLIPNeRFTrainer(cfg)

    # 创建logger
    logger = WandbLogger(project='clip-nerf')

    # 创建训练器

    # 创建检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',  # 保存检查点的目录
        filename='model-{epoch:02d}-{val_loss:.2f}',  # 文件名格式
        monitor='val_loss',  # 监控的指标
        mode='min',  # 模式：'min' 或 'max'
        save_top_k=3,  # 保存最好的 k 个模型
        save_last=True,  # 是否保存最后一个epoch的模型
        every_n_epochs=1  # 每隔多少个epoch保存一次
    )

    pl_trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=32,  # 使用 16 位混合精度训练
        gradient_clip_val=1.0,  # 在训练器级别设置梯度裁剪
        max_epochs=cfg.max_epochs,
        gpus=2,
        callbacks=[checkpoint_callback],
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
