# utils/metrics.py
import torch
import clip
import numpy as np
# from pytorch3d.renderer import Meshes


class Metrics:
    def __init__(self, device='cuda'):
        self.device = device
        # 加载CLIP模型
        self.clip_model, _ = clip.load("ViT-B/32", device=device)

        # 冻结CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def compute_chamfer_distance(self, pred_points, target_points):
        """计算Chamfer距离"""
        from pytorch3d.loss import chamfer_distance

        loss, _ = chamfer_distance(pred_points, target_points)
        return loss.item()

    def compute_clip_sim(self, rendered_images, text_prompts):
        """
        计算CLIP-SIM评估指标

        Args:
            rendered_images: 渲染的图像 [B, 3, H, W]
            text_prompts: 文本提示列表，长度为B

        Returns:
            clip_sim: CLIP相似度 [0,1]之间的标量
        """
        # 确保图像尺寸正确
        if rendered_images.shape[2:] != (224, 224):
            rendered_images = torch.nn.functional.interpolate(
                rendered_images,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )

        # 编码图像
        with torch.no_grad():
            image_features = self.clip_model.encode_image(rendered_images)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # 编码文本
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # 计算余弦相似度
        similarities = (image_features * text_features).sum(dim=1)

        # 平均相似度
        clip_sim = similarities.mean().item()

        return clip_sim

    def evaluate(self, model, dataloader, renderer):
        """
        评估模型性能

        Args:
            model: 待评估的模型
            dataloader: 数据加载器
            renderer: 渲染器

        Returns:
            metrics: 包含各评估指标的字典
        """
        model.eval()
        chamfer_distances = []
        clip_sims = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                target_points = batch['points'].to(self.device)
                text_prompts = batch['text_prompts']

                # 生成网格
                meshes = model.generate_mesh(images, is_training=False)

                # 提取点云
                pred_points = meshes.verts_padded()

                # 计算Chamfer距离
                cd = self.compute_chamfer_distance(pred_points, target_points)
                chamfer_distances.append(cd)

                # 渲染网格
                rendered_images = renderer(meshes)
                rendered_images = rendered_images[..., :3].permute(0, 3, 1, 2)

                # 计算CLIP-SIM
                clip_sim = self.compute_clip_sim(rendered_images, text_prompts)
                clip_sims.append(clip_sim)

        # 计算平均指标
        avg_cd = np.mean(chamfer_distances)
        avg_clip_sim = np.mean(clip_sims)

        return {
            'chamfer_distance': avg_cd,
            'clip_sim': avg_clip_sim
        }
