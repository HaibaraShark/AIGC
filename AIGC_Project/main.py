import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from models.diffusion_model import UNet, DiffusionModel
from models.style_transfer import StyleTransfer
from models.image_optimizer import ImageOptimizer
from models.pattern_generator import PatternGenerator

class AIGCApplication:
    def __init__(self):
        # 初始化各模块
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 扩散模型
        unet = UNet().to(self.device)
        self.diffusion = DiffusionModel(unet)
        
        # 风格迁移
        self.style_transfer = StyleTransfer()
        
        # 图像优化
        self.optimizer = ImageOptimizer()
        
        # 纹样生成
        self.pattern_generator = PatternGenerator()
        
        # 图像转换
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        self.inv_transform = transforms.ToPILImage()
    
    def generate_image(self, style='chinese', size=(64, 64)):
        """生成国风图像"""
        # 生成基础图像
        batch_size = 1
        generated = self.diffusion.sample(batch_size, size[0])
        generated = generated.clamp(0, 1)
        
        # 转换为PIL图像
        image = self.inv_transform(generated[0])
        
        # 风格迁移（如果需要）
        if style == 'chinese':
            # 这里可以添加风格图像进行风格迁移
            # 暂时返回原始生成的图像
            pass
        
        return image
    
    def optimize_image(self, image):
        """优化图像"""
        # 转换为张量
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 超分
        sr_image = self.optimizer.super_resolve(tensor)
        sr_image = sr_image.clamp(0, 1)
        
        # 边缘增强
        enhanced = self.optimizer.enhance_edges(sr_image[0])
        
        # 转换为PIL图像
        optimized = self.inv_transform(enhanced)
        return optimized
    
    def generate_pattern(self, style='chinese', size=(128, 128)):
        """生成纹样"""
        pattern = self.pattern_generator.generate_pattern(style, size)
        seamless = self.pattern_generator.make_seamless(pattern)
        
        # 转换为PIL图像
        pattern_image = self.inv_transform(seamless)
        return pattern_image
    
    def segment_image(self, image):
        """分割图像"""
        # 转换为张量
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 分割
        mask = self.pattern_generator.segment(tensor)
        mask = mask.squeeze(0)
        
        # 转换为PIL图像
        mask_image = self.inv_transform(mask)
        return mask_image

if __name__ == "__main__":
    # 初始化应用
    app = AIGCApplication()
    
    # 生成国风图像
    print("生成国风图像...")
    generated_image = app.generate_image()
    generated_image.save("generated_image.png")
    print("生成图像已保存为 generated_image.png")
    
    # 优化图像
    print("优化图像...")
    optimized_image = app.optimize_image(generated_image)
    optimized_image.save("optimized_image.png")
    print("优化图像已保存为 optimized_image.png")
    
    # 生成纹样
    print("生成纹样...")
    pattern = app.generate_pattern()
    pattern.save("pattern.png")
    print("纹样已保存为 pattern.png")
    
    # 分割图像
    print("分割图像...")
    mask = app.segment_image(generated_image)
    mask.save("segment_mask.png")
    print("分割掩码已保存为 segment_mask.png")
    
    print("所有任务完成！")