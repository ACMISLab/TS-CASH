import inspect
from functools import wraps


class Singleton(type):
    """
    在 @dataclass 中实现单例模式

    @dataclass
    class MyClass(metaclass=Singleton):
        data: str
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class ClassUtil:

    @staticmethod
    def get_instance_attributes_as_json_str(instance_):
        """
        returns the instance attributes as json string:

        return:
            {'data_id': 'NAB_data_CloudWatch_1.out', 'data_sample_method': 'random', 'data_sample_rate': 0.001953125, 'dataset_name': 'NAB', 'exp_index': 1, 'exp_total': 6, 'model_name': 'pca', 'window_size': 64}

        for class:
        class ExpConf:
            def __init__(self, model_name,
                         dataset_name,
                         data_id,
                         data_sample_method,
                         data_sample_rate,
                         exp_index,
                         exp_total):

                self.window_size = 64
                self.model_name = model_name
                self.data_sample_method = data_sample_method
                self.data_sample_rate = data_sample_rate
                self.dataset_name = dataset_name
                self.data_id = data_id
                self.exp_total = exp_total
                self.exp_index = exp_index

        Parameters
        ----------
        instance_ : a class instance
            e.g.
            ClassUtil.get_instance_attributes_as_json_str(ExpConf())

        Returns
        -------

        """
        ret = {}
        for k, v in inspect.getmembers(instance_, lambda a: not (inspect.isroutine(a))):
            if not str(k).startswith("__"):
                ret[k] = v
        return ret

    @staticmethod
    def get_class_attribute_names(target_class: object) -> list:
        """
        获取目标类中所有的静态属性名称

        Parameters
        ----------
        target_class :

        Returns
        -------

        """
        all_attributes = dir(target_class)
        return [getattr(target_class, attr) for attr in all_attributes if
                not callable(getattr(target_class, attr)) and not attr.startswith("__")]


def initializer(func):
    """
    Decorator for the __init__ method.
    Automatically assigns the parameters.
    """
    argspec = inspect.getfullargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(argspec.args[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)
        for name, default in zip(reversed(argspec.args), reversed(argspec.defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)
        func(self, *args, **kargs)

    return wrapper
