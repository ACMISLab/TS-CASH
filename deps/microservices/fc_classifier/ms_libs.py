import torch
def get_device():
    """获取加速设备，包括cuda、cpu，返回字符串"""
    # DGL库不支持MPS设备，所以跳过MPS检查
    if torch.cuda.is_available():
        return "gpu"
    else:
        return "cpu"

if __name__ == "__main__":
    print(get_device())
