import torch
import torch.nn as nn
import cv2
import numpy as np

class SuperResolution(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        
        # 简单的超分网络
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3 * scale_factor * scale_factor, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )
    
    def forward(self, x):
        return self.model(x)

class ImageOptimizer:
    def __init__(self):
        self.sr_model = SuperResolution().to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def super_resolve(self, image):
        """图像超分"""
        # 转换为张量
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensor = image.to(device)
        
        # 超分
        with torch.no_grad():
            output = self.sr_model(tensor)
        
        return output
    
    def inpaint(self, image, mask):
        """图像修复"""
        # 转换为numpy数组
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
        
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().transpose(1, 2, 0)
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = mask
        
        # 使用OpenCV的inpaint函数
        result = cv2.inpaint(image_np, mask_np, 3, cv2.INPAINT_TELEA)
        
        # 转换回张量
        result_tensor = torch.from_numpy(result.transpose(2, 0, 1)).float() / 255.0
        return result_tensor
    
    def enhance_edges(self, image):
        """边缘增强"""
        # 转换为numpy数组
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
        
        # 边缘检测
        edges = cv2.Canny(image_np, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 边缘增强
        result = cv2.addWeighted(image_np, 0.8, edges, 0.2, 0)
        
        # 转换回张量
        result_tensor = torch.from_numpy(result.transpose(2, 0, 1)).float() / 255.0
        return result_tensor