# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from models.base_model import SingleViewReconstructor
from models.clip_encoder import CLIPEncoder
from losses.combined_loss import CombinedLoss
from data.shapenet import ShapeNetDataset


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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # 提取批次数据
        images = batch['image']
        text = batch['text']
        gt_depth = batch.get('depth')
        gt_mesh = batch.get('mesh')

        # 前向传播
        pred_mesh, feature_volume = self.model(images)

        # CLIP特征编码
        image_features = self.clip_encoder.encode_image(images)
        text_features = self.clip_encoder.encode_text(text)

        # 准备模型输出和目标
        outputs = {
            'pred_mesh': pred_mesh,
            'feature_volume': feature_volume,
            'image_features': image_features,
            'model_params': self.model.parameters(),
            # 这里需要实现一个渲染函数来获取渲染图像和深度
            'rendered_images': self._render_mesh(pred_mesh),
            'pred_depth': self._render_depth(pred_mesh)
        }

        targets = {
            'gt_depth': gt_depth,
            'gt_mesh': gt_mesh,
            'text_features': text_features,
            'visibility_mask': batch.get('mask', torch.ones_like(gt_depth))
        }

        # 计算损失
        loss_dict = self.loss_fn(outputs, targets)

        # 记录损失
        for k, v in loss_dict.items():
            self.log(f'train/{k}_loss', v)

        return loss_dict['total']

    def _render_mesh(self, mesh):
        """渲染网格为RGB图像"""
        # 使用PyTorch3D的渲染器
        # 这里简化实现，实际需要设置相机、光照等
        from pytorch3d.renderer import (
            look_at_view_transform,
            FoVPerspectiveCameras,
            PointLights,
            RasterizationSettings,
            MeshRenderer,
            MeshRasterizer,
            SoftPhongShader,
            TexturesVertex
        )

        # 设置相机
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = FoVPerspectiveCameras(device=mesh.device, R=R, T=T)

        # 设置光照
        lights = PointLights(device=mesh.device, location=[[0.0, 0.0, -3.0]])

        # 设置渲染器
        raster_settings = RasterizationSettings(
            image_size=128,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # 创建渲染器
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=mesh.device, cameras=cameras, lights=lights)
        )

        # 添加纹理（简化为顶点颜色）
        verts_rgb = torch.ones_like(mesh.verts_padded())  # 白色
        textures = TexturesVertex(verts_features=verts_rgb)
        mesh.textures = textures

        # 渲染
        images = renderer(mesh)

        return images

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
        # 使用Adam优化器
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr)

        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.max_epochs
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train/total_loss'
        }

    def train_dataloader(self):
        # 创建ShapeNet数据集
        dataset = ShapeNetDataset(
            root=self.cfg.data_root,
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
            pin_memory=True
        )

        return loader


# 主函数
def main():
    # 配置参数
    cfg = {
        'data_root': './data/shapenet',
        'categories': ['chair', 'table', 'car'],
        'batch_size': 16,
        'num_workers': 4,
        'lr': 1e-4,
        'max_epochs': 100,
        'lambda_depth': 0.5,
        'lambda_clip': 0.3,
        'lambda_edge': 0.2,
        'lambda_reg': 0.01,
        'lambda_lap': 0.1,
        'beta_grad': 0.05,
        'temperature': 0.07,
        'clip_model': 'ViT-B/32'
    }

    # 创建训练器
    trainer = CLIPNeRFTrainer(cfg)

    # 创建logger
    logger = WandbLogger(project='clip-nerf')

    # 创建训练器
    pl_trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        gpus=1,
        logger=logger,
        precision=16,  # 混合精度训练
        gradient_clip_val=1.0
    )

    # 开始训练
    pl_trainer.fit(trainer)


if __name__ == '__main__':
    main()
