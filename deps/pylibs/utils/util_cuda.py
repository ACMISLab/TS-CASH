import sys
import nni
import os


def nni_cuda_visible_devices():
    """
    Set the CUDA_VISIBLE_DEVICES=nni.get_sequence_id() % n_gpus.
    Note: this function must be called in the head of a *.py

    Returns
    -------

    """
    device = 'cpu'
    if '--gpus' in sys.argv:
        assert 'torch' not in sys.modules, 'function nni_cuda_visible_devices must call before import torch'
        try:
            avaliable_gpus = sys.argv[sys.argv.index("--gpus") + 1].split(",")
        except IndexError:
            raise ValueError("The values of --gpus must be specified")
        n_gpus = len(avaliable_gpus)
        if n_gpus > 0:
            gpu_id = nni.get_sequence_id() % n_gpus
            os.environ['CUDA_VISIBLE_DEVICES'] = str(avaliable_gpus[gpu_id])
            device = 'cuda'
    return device
