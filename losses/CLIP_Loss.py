# losses/clip_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIPLoss(nn.Module):
    def __init__(self, clip_model="ViT-B/32", temperature=0.07, device='cuda'):
        super().__init__()
        # print(f"Initializing CLIP Loss with model: {clip_model}")
        try:
            self.model, self.preprocess = clip.load(clip_model, device=device)
            # print(f"CLIP model loaded successfully. Output dim: {self.model.visual.output_dim}")
        except Exception as e:
            # print(f"Error loading CLIP model: {e}")
            raise e
        self.model = self.model.float()

        self.temperature = temperature

        # 冻结CLIP参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 密度场到CLIP空间的投影网络
        self.density_projector = nn.Sequential(
            nn.Linear(512, 512),  # 假设密度特征维度为512
            nn.ReLU(),
            nn.Linear(512, self.model.visual.output_dim)
        )
        # print(f"Density projector initialized: {self.density_projector}")

    def forward(self, density_features, text_prompts, images=None):
        """
        计算CLIP对比学习损失

        Args:
            density_features: 密度场特征 [B, C]
            text_prompts: 文本提示列表，长度为B
            images: 原始输入图像 [B, 3, H, W]，可选

        Returns:
            loss: CLIP对比学习损失
        """
        # 详细记录输入
        print(f"\n=== CLIP Loss Debug ===")
        print(
            f"Density features: {type(density_features)}, shape: {density_features.shape if density_features is not None else 'None'}")
        print(f"Text prompts: {type(text_prompts)}, content: {text_prompts}")
        print(f"Images: {type(images)}, shape: {images.shape if images is not None else 'None'}")

        if density_features is None or text_prompts is None:
            print("CLIP Loss: Missing inputs - returning zero loss")
            return torch.tensor(0.0, device="cuda")

        if isinstance(text_prompts, list) and len(text_prompts) == 0:
            print("CLIP Loss: Empty text prompts list - returning zero loss")
            return torch.tensor(0.0, device="cuda")

        try:
            # 将密度特征投影到CLIP空间
            density_features = self.density_projector(density_features)
            print(f"Projected density features shape: {density_features.shape}")

            density_features = density_features / density_features.norm(dim=1, keepdim=True)
            print(f"Normalized density features shape: {density_features.shape}")

            # 编码文本
            print(f"Tokenizing text prompts: {text_prompts}")
            text_tokens = clip.tokenize(text_prompts).to(density_features.device)
            print(f"Text tokens shape: {text_tokens.shape}")

            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                print(f"Text features shape: {text_features.shape}")

                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                print(f"Normalized text features shape: {text_features.shape}")

            # 计算相似度
            logits = (density_features @ text_features.T) / self.temperature
            print(f"Logits shape: {logits.shape}, values: {logits[:2, :2]}")

            # 对比损失（每个密度特征与其对应的文本匹配）
            labels = torch.arange(len(density_features), device=density_features.device)
            print(f"Labels: {labels}")

            loss_1 = F.cross_entropy(logits, labels)
            loss_2 = F.cross_entropy(logits.T, labels)
            print(f"Loss components: {loss_1.item()}, {loss_2.item()}")

            loss = (loss_1 + loss_2) / 2
            print(f"Combined loss: {loss.item()}")

            # 如果提供了原始图像，可以添加三元组损失
            if images is not None:
                # 确保图像尺寸正确
                if images.shape[2:] != (224, 224):
                    print(f"Resizing images from {images.shape[2:]} to (224, 224)")
                    images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

                # 提取图像特征
                with torch.no_grad():
                    image_features = self.model.encode_image(images)
                    print(f"Image features shape: {image_features.shape}")

                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    print(f"Normalized image features shape: {image_features.shape}")

                # 计算密度特征与图像特征的一致性损失
                consistency_loss = 1.0 - F.cosine_similarity(density_features, image_features).mean()
                print(f"Consistency loss: {consistency_loss.item()}")

                # 添加到总损失
                loss = loss + 0.5 * consistency_loss
                print(f"Final loss with consistency: {loss.item()}")

            print(f"=== CLIP Loss: {loss.item()} ===\n")
            return loss

        except Exception as e:
            print(f"Error in CLIP loss calculation: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device="cuda")


class ImprovedCLIPLoss(nn.Module):
    def __init__(self, clip_model="ViT-B/32", device='cuda'):
        super().__init__()
        self.model, self.preprocess = clip.load(clip_model, device=device)

        # 冻结CLIP参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 不再需要密度投影网络
        # self.density_projector = ...

        # 图像大小
        self.image_size = 224  # CLIP 标准输入尺寸

    def forward(self, points, densities, text_prompts, normals=None, original_images=None):
        """
        改进的CLIP损失，使用点云渲染为图像的方式

        Args:
            points: 点云坐标 [B, N, 3]
            densities: 密度值 [B, N, 1]
            text_prompts: 文本提示列表
            normals: 点云法线 [B, N, 3]，可选
            original_images: 原始输入图像 [B, 3, H, W]，可选
        """
        if points is None or text_prompts is None:
            return torch.tensor(0.0, device="cuda")

        try:
            # 1. 生成点云的多种视图表示
            point_images = self._generate_point_cloud_views(points, densities, normals)

            # 2. 编码图像和文本
            with torch.no_grad():
                # 编码点云图像
                image_features = self.model.encode_image(point_images)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

                # 编码文本
                text_tokens = clip.tokenize(text_prompts).to(points.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # 如果有原始图像，也编码它们
                if original_images is not None:
                    # 调整大小以适应CLIP
                    if original_images.shape[2:] != (self.image_size, self.image_size):
                        original_images = F.interpolate(
                            original_images,
                            size=(self.image_size, self.image_size),
                            mode='bilinear'
                        )
                    orig_image_features = self.model.encode_image(original_images)
                    orig_image_features = orig_image_features / orig_image_features.norm(dim=1, keepdim=True)

            # 3. 计算点云图像与文本的对比损失
            batch_size = points.shape[0]
            labels = torch.arange(batch_size, device=points.device)

            # 计算相似度矩阵
            logits = 100.0 * image_features @ text_features.T

            # 对比损失
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            contrastive_loss = (loss_i2t + loss_t2i) / 2.0

            # 4. 如果有原始图像，添加一致性损失
            consistency_loss = 0.0
            if original_images is not None:
                # 点云图像与原始图像的一致性
                consistency_loss = 1.0 - F.cosine_similarity(
                    image_features, orig_image_features
                ).mean()

            # 5. 组合损失
            total_loss = contrastive_loss + 0.5 * consistency_loss

            return total_loss

        except Exception as e:
            print(f"Error in CLIP loss calculation: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=points.device)

    def _generate_point_cloud_views(self, points, densities, normals=None):
        """
        生成点云的多视角表示

        返回: 点云的图像表示 [B*V, 3, 224, 224]，其中V是视角数
        """
        batch_size = points.shape[0]
        device = points.device

        # 1. 生成深度图和法线图
        depth_maps = self._points_to_depth_map(points, densities)  # [B, 1, H, W]

        if normals is not None:
            normal_maps = self._normals_to_normal_map(points, normals, densities)  # [B, 3, H, W]
        else:
            # 从深度图估计法线
            normal_maps = self._estimate_normals_from_depth(depth_maps)  # [B, 3, H, W]

        # 2. 组合为RGB图像
        # 使用深度作为一个通道，法线的x,y作为其他通道
        combined_images = torch.cat([
            depth_maps,  # [B, 1, H, W]
            normal_maps[:, :2, :, :],  # [B, 2, H, W]
        ], dim=1)

        # 3. 归一化到[0,1]范围
        combined_images = (combined_images - combined_images.min()) / (
                    combined_images.max() - combined_images.min() + 1e-8)

        # 4. 转换为RGB格式
        # 这样CLIP可以理解这些图像，因为它是在RGB图像上训练的
        rgb_images = combined_images  # [B, 3, H, W]

        return rgb_images

    def _points_to_depth_map(self, points, densities, size=224):
        """将点云转换为深度图"""
        batch_size, num_points, _ = points.shape
        device = points.device

        # 初始化深度图
        depth_maps = torch.zeros(batch_size, 1, size, size, device=device)
        weight_maps = torch.zeros(batch_size, 1, size, size, device=device)

        # 使用sigmoid激活确保权重在[0,1]范围内
        weights = torch.sigmoid(densities).squeeze(-1)  # [B, N]

        for b in range(batch_size):
            # 提取坐标
            x = points[b, :, 0]  # 假设x轴对应图像的宽度
            y = points[b, :, 1]  # 假设y轴对应图像的高度
            z = points[b, :, 2]  # 深度值

            # 归一化x,y坐标到[0,1]范围
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()

            x_norm = (x - x_min) / (x_max - x_min + 1e-8)
            y_norm = (y - y_min) / (y_max - y_min + 1e-8)

            # 转换为像素坐标
            x_px = (x_norm * (size - 1)).long().clamp(0, size - 1)
            y_px = (y_norm * (size - 1)).long().clamp(0, size - 1)

            # 归一化深度值
            z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)

            # 填充深度图
            for i in range(num_points):
                xi, yi = x_px[i], y_px[i]
                zi, wi = z_norm[i], weights[b, i]

                depth_maps[b, 0, yi, xi] += zi * wi
                weight_maps[b, 0, yi, xi] += wi

            # 计算加权平均深度
            valid_mask = weight_maps[b] > 0
            depth_maps[b][valid_mask] = depth_maps[b][valid_mask] / weight_maps[b][valid_mask]

        return depth_maps

    def _normals_to_normal_map(self, points, normals, densities, size=224):
        """将点云法线转换为法线图"""
        batch_size, num_points, _ = points.shape
        device = points.device

        # 初始化法线图
        normal_maps = torch.zeros(batch_size, 3, size, size, device=device)
        weight_maps = torch.zeros(batch_size, 1, size, size, device=device)

        # 使用sigmoid激活确保权重在[0,1]范围内
        weights = torch.sigmoid(densities).squeeze(-1)  # [B, N]

        for b in range(batch_size):
            # 提取坐标和法线
            x = points[b, :, 0]
            y = points[b, :, 1]
            nx = normals[b, :, 0]
            ny = normals[b, :, 1]
            nz = normals[b, :, 2]

            # 归一化x,y坐标到[0,1]范围
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()

            x_norm = (x - x_min) / (x_max - x_min + 1e-8)
            y_norm = (y - y_min) / (y_max - y_min + 1e-8)

            # 转换为像素坐标
            x_px = (x_norm * (size - 1)).long().clamp(0, size - 1)
            y_px = (y_norm * (size - 1)).long().clamp(0, size - 1)

            # 填充法线图
            for i in range(num_points):
                xi, yi = x_px[i], y_px[i]
                nxi, nyi, nzi = nx[i], ny[i], nz[i]
                wi = weights[b, i]

                normal_maps[b, 0, yi, xi] += nxi * wi  # R通道存储x分量
                normal_maps[b, 1, yi, xi] += nyi * wi  # G通道存储y分量
                normal_maps[b, 2, yi, xi] += nzi * wi  # B通道存储z分量
                weight_maps[b, 0, yi, xi] += wi

            # 计算加权平均法线 - 修复维度不匹配问题
            valid_mask = weight_maps[b, 0] > 0  # 形状为 [H, W]

            # 方法1：使用广播机制
            for c in range(3):
                # 创建一个布尔掩码，与normal_maps[b, c]形状相同
                channel_mask = valid_mask  # [H, W]

                # 使用布尔索引
                normal_maps[b, c][channel_mask] = normal_maps[b, c][channel_mask] / weight_maps[b, 0][channel_mask]

            # 归一化法线向量
            norm = torch.sqrt(torch.sum(normal_maps[b] ** 2, dim=0, keepdim=True) + 1e-8)
            normal_maps[b] = normal_maps[b] / norm

            # 将法线值从[-1,1]映射到[0,1]范围
            normal_maps[b] = (normal_maps[b] + 1) / 2

        return normal_maps

    def _estimate_normals_from_depth(self, depth_maps):
        """从深度图估计法线图"""
        batch_size, _, height, width = depth_maps.shape
        device = depth_maps.device

        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3,
                                                                                                              3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3,
                                                                                                              3)

        # 计算深度梯度
        dx = F.conv2d(depth_maps, sobel_x, padding=1)
        dy = F.conv2d(depth_maps, sobel_y, padding=1)

        # 创建法线向量 [B, 3, H, W]
        normals = torch.cat([-dx, -dy, torch.ones_like(dx)], dim=1)

        # 归一化法线
        norm = torch.sqrt(torch.sum(normals ** 2, dim=1, keepdim=True))
        normals = normals / (norm + 1e-8)

        # 将法线值从[-1,1]映射到[0,1]范围
        normals = (normals + 1) / 2

        return normals
