import torch
import torch.nn as nn
import cv2
import numpy as np

class SimpleSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        # 简单的分割网络
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class PatternGenerator:
    def __init__(self):
        self.segmentation_model = SimpleSegmentation().to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def segment(self, image):
        """图像分割"""
        # 转换为张量
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensor = image.to(device)
        
        # 分割
        with torch.no_grad():
            mask = self.segmentation_model(tensor)
        
        # 二值化
        mask = (mask > 0.5).float()
        return mask
    
    def generate_pattern(self, style, size=(128, 128)):
        """生成重复纹样"""
        pattern = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        
        if style == 'geometric':
            # 生成几何纹样
            for i in range(0, size[0], 32):
                for j in range(0, size[1], 32):
                    color = np.random.randint(0, 255, 3)
                    cv2.rectangle(pattern, (i, j), (i+30, j+30), color.tolist(), -1)
        elif style == 'floral':
            # 生成花卉纹样
            for i in range(0, size[0], 40):
                for j in range(0, size[1], 40):
                    color = np.array([255, 0, 0])
                    cv2.circle(pattern, (i+20, j+20), 15, color.tolist(), -1)
        elif style == 'chinese':
            # 生成中国传统纹样
            for i in range(0, size[0], 48):
                for j in range(0, size[1], 48):
                    color = np.array([255, 215, 0])
                    cv2.rectangle(pattern, (i, j), (i+46, j+46), color.tolist(), 2)
                    cv2.line(pattern, (i, j), (i+46, j+46), color.tolist(), 2)
                    cv2.line(pattern, (i+46, j), (i, j+46), color.tolist(), 2)
        
        # 转换为张量
        pattern_tensor = torch.from_numpy(pattern.transpose(2, 0, 1)).float() / 255.0
        return pattern_tensor
    
    def make_seamless(self, pattern):
        """制作无缝纹样"""
        # 转换为numpy数组
        if isinstance(pattern, torch.Tensor):
            pattern_np = pattern.cpu().numpy().transpose(1, 2, 0)
            pattern_np = (pattern_np * 255).astype(np.uint8)
        else:
            pattern_np = pattern
        
        # 水平无缝
        pattern_np[:, -10:] = pattern_np[:, :10]
        # 垂直无缝
        pattern_np[-10:, :] = pattern_np[:10, :]
        
        # 转换回张量
        pattern_tensor = torch.from_numpy(pattern_np.transpose(2, 0, 1)).float() / 255.0
        return pattern_tensor