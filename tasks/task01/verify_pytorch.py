import torch
import sys
def verify():
    """
    ⼀个⽤于验证 PyTorch GPU 环境是否配置成功的函数。
    """
    print("--- PyTorch 环境验证开始 ---")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    # 核⼼检查：CUDA 是否被 PyTorch 检测到
    is_cuda_available = torch.cuda.is_available()
    print(f"\nCUDA 是否可⽤: {'是' if is_cuda_available else '否'}")
    if is_cuda_available:
        # 如果可⽤，则打印更详细的 GPU 信息
        gpu_count = torch.cuda.device_count()
        print(f"检测到的 GPU 数量: {gpu_count}")
        current_gpu_id = torch.cuda.current_device()
        current_gpu_name = torch.cuda.get_device_name(current_gpu_id)
        print(f"当前默认 GPU (ID: {current_gpu_id}): {current_gpu_name}")
        # 创建⼀个简单的张量并移动到 GPU
        try:
            tensor_cpu = torch.randn(3, 3)
            print(f"\n在 CPU 上创建了⼀个tensor: \n{tensor_cpu}")
            tensor_gpu = tensor_cpu.to("cuda")
            print(f"已成功将tensor移动到 GPU ({tensor_gpu.device}): \n{tensor_gpu}")
            print("\n你的 PyTorch GPU 环境已成功配置！")
        except Exception as e:
            print(f"在尝试使⽤ GPU 时发⽣错误: {e}")
    else:
        # 如果不可⽤，提供排查建议
        print("\n[排查建议]:")
        print("1. 确认 NVIDIA 驱动已正确安装 (可运⾏ `nvidia-smi` 命令检查)。")
        print("2. 确认已安装了与 PyTorch 版本兼容的 CUDA Toolkit。")
        print("3. 确认你安装 PyTorch 时，使⽤的是带有 CUDA 后缀的命令 (例如, ...whl/cu118)，⽽不是 CPU 版本。")
        print(" 如果安装错误，请先卸载 (`pip uninstall torch`) 再重新安装正确的版本。")
        print("--- 验证结束 ---")
if __name__ == "__main__":
    verify()