__author__ = "gsunwu@163.com"

import os.path
from importlib.util import spec_from_file_location, module_from_spec

dir_crt = os.path.dirname(__file__)


def get_feature_select_method(class_name, *args, **kwargs):
    # 动态导入模块
    # 获取模块名称
    file_name = os.path.join(dir_crt, f"{class_name}.py")
    module_name = os.path.splitext(os.path.basename(file_name))[0]
    # 动态加载模块
    spec = spec_from_file_location(module_name, file_name)
    module =  module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取类
    cls = getattr(module, class_name)

    # 实例化类
    instance = cls(*args, **kwargs)

    return instance
