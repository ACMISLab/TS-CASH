# import json
# import logging
# import sys
# import time
# import traceback
# from dataclasses import dataclass
# import requests
# from requests.auth import HTTPBasicAuth
#
#
# @dataclass
# class Vector:
#     # http://<ip>:<port>
#     # [{"level":"info","job":"test","log":"test message for openobserve"}]
#     LOG_FORMAT_PLAIN = '[%(filename)s:%(lineno)d] %(levelname)s - %(message)s'
#     LOG_FORMAT_JSON = '[{"level":"%(levelname)s","msg":"%(message)s","identifier":"%(filename)s:%(lineno)d"}]'
#     logger = None
#
#     @staticmethod
#     def get_logger(name=None, level=logging.INFO) -> logging.Logger:
#         """
#         将 logging 的输出设置为 Vector
#
#         docker pull timberio/vector:0.36.0-debian
#         Returns
#         -------
#
#         """
#         # 配置 OpenObserver的 http 日志
#         if Vector.logger is not None:
#             pass
#         else:
#             logger = logging.getLogger(name)
#             # http_handler = OpenObserveHandler()
#             # formatter = logging.Formatter(Vector.LOG_FORMAT_JSON)
#             # http_handler.setFormatter(formatter)
#             # logger.addHandler(http_handler)
#
#             # 配置控制台日志
#             stream_handler = logging.StreamHandler(stream=sys.stdout)
#             stream_handler.setFormatter(logging.Formatter(Vector.LOG_FORMAT_PLAIN))
#             logger.addHandler(stream_handler)
#
#             # 日志等级
#             logger.setLevel(level)
#             Vector.logger = logger
#
#         return Vector.logger
