import torch
from main import AIGCApplication

print("测试AIGC应用...")

# 检查是否有GPU
print(f"使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# 初始化应用
app = AIGCApplication()
print("应用初始化成功!")

# 测试生成图像
print("\n测试生成图像...")
try:
    generated_image = app.generate_image()
    print("生成图像成功!")
    generated_image.save("test_generated.png")
    print("生成图像已保存为 test_generated.png")
except Exception as e:
    print(f"生成图像失败: {e}")

# 测试优化图像
print("\n测试优化图像...")
try:
    optimized_image = app.optimize_image(generated_image)
    print("优化图像成功!")
    optimized_image.save("test_optimized.png")
    print("优化图像已保存为 test_optimized.png")
except Exception as e:
    print(f"优化图像失败: {e}")

# 测试生成纹样
print("\n测试生成纹样...")
try:
    pattern = app.generate_pattern()
    print("生成纹样成功!")
    pattern.save("test_pattern.png")
    print("纹样已保存为 test_pattern.png")
except Exception as e:
    print(f"生成纹样失败: {e}")

# 测试分割图像
print("\n测试分割图像...")
try:
    mask = app.segment_image(generated_image)
    print("分割图像成功!")
    mask.save("test_mask.png")
    print("分割掩码已保存为 test_mask.png")
except Exception as e:
    print(f"分割图像失败: {e}")

print("\n测试完成!")