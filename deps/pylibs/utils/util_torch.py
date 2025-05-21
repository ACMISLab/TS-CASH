import torch


def is_mps_available():
    """
    检查mps 是否可用
    Returns
    -------

    """
    return torch.backends.mps.is_available()


if __name__ == '__main__':
    print(is_mps_available())
