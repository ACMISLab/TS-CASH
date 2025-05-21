#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/18 15:21
# @Author  : gsunwu@163.com
# @File    : util_tailscale.py
# @Description:
from pylibs.utils.util_bash import BashUtil
from pylibs.utils.util_system import UtilSys


# @DeprecationWarning
# def run_dask_worker_tailscale(
#         port=DaskENV.SERVER_PORT,
#         scheduler="tcp://your_server_ip:9300",
#         process_name=None,
#         cuda_visible_devices=0,
# ):
#     """
#
#     Parameters
#     ----------
#     port : int
#         进程工作端口
#     n_workers : int
#         工作进程的数量
#     scheduler : str
#         scheduler 的地址
#
#     Returns
#     -------
#
#     """
#
#     if is_port_listing(port):
#         print(f"Port {port} is in using")
#         wait_kill_process_on_port(port=port)
#     else:
#         print(f"Start worker on port={port}")
#
#     local_tailscale_ip = UtilTailscale.get_local_ip()
#     while True:
#         if local_tailscale_ip.startswith("100"):
#             break
#         local_tailscale_ip = UtilTailscale.get_local_ip()
#         time.sleep(1)
#
#     assert local_tailscale_ip is not None
#     # 本机监听地址
#     listen_address = f"tcp://your_server_ip:{port}"
#
#     # scheduler 访问的地址，需要保证 scheduler 能够直连这个地址
#     contact_address = f"tcp://{local_tailscale_ip}:{port}"
#
#     # CUDA_VISIBLE_DEVICES=7 TF_FORCE_GPU_ALLOW_GROWTH=true  dask cuda worker --pid-file /tmp/daskworker-g7t1.pid   --memory-limit '11GB' --device-memory-limit "3500M" --name daskworker-g7t1  --nthreads 1 tcp://your_server_ip:6006
#     target_run = partial(main_worker,
#                          listen_address=listen_address,
#                          contact_address=contact_address,
#                          nthreads=1,
#                          scheduler=scheduler,
#                          memory_limit="11GB",
#                          name=process_name,
#                          cuda_visible_devices=cuda_visible_devices
#                          )
#     print(f"Start worker [{process_name}] at {port}")
#     target_run()
#     # daemon_run(target_run)
class UtilTailscale:
    @staticmethod
    def get_local_ip():
        if UtilSys.is_macos():
            return BashUtil.exe_cmd("/Applications/Tailscale.app/Contents/MacOS/Tailscale ip").strip()
        else:
            return BashUtil.exe_cmd("tailscale ip").strip()


if __name__ == '__main__':
    print(UtilTailscale.get_local_ip())
