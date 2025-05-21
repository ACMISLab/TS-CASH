from enum import Enum, unique
from nni.common.hpo_utils import format_search_space

from tuner.sampling.lhs_sampling import LHSSampling
from tuner.base_tuner import BaseTuner
from pylibs.utils.util_log import get_logger

log = get_logger()


@unique
class OptimizeMode(Enum):
    Minimize = 'minimize'
    Maximize = 'maximize'


class LHSRNSTuner(BaseTuner):
    def __init__(self, *args, **kwargs):
        # Init parameter
        super().__init__(args, kwargs)

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''
        Receive trial's final result.
        parameter_id: int
        parameters: object created by 'generate_parameters()'
        value: final metrics of the trial, including default metric.


        E.g.
            parameter_id:1,
            parameters:{'latent_dim': 5.0},
            value:1,
            **kwargs:{'customized': False, 'trial_job_id': 'lJSJ9'}

        value: float or dict

        value['default']
        '''

        # your code implements here.

        metric = self.check_receive_metric(parameters, value)
        self.trial_results.append({
            "parameter_id": parameter_id,
            "metric": metric,
            "parameters": parameters
        })

    def generate_parameters(self, parameter_id, **kwargs):
        '''
        Returns a set of trial (hyper-)parameters, as a serializable object
        parameter_id: int
        '''
        # your code implements here.

        your_parameters = self.get_parameters(parameter_id)
        for val in self.format_search_space.values():
            if val.type == "choice":
                your_parameters[val.name] = val.values[int(your_parameters[val.name])]
        return your_parameters

    def update_search_space(self, search_space):
        '''
        Tuners are advised to support updating search space at run-time.
        If a nni_tuner can only set search space once before generating first hyper-parameters,
        it should explicitly document this behaviour.
        cur_search_space_: JSON object created by experiment owner, e.g.:
        {'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]}}
        '''
        # your code implements here.
        self.format_search_space = format_search_space(search_space)
        self.search_space = search_space

        sampler = LHSSampling(self.round_sample_size[0], self.search_space)
        init_samples = sampler.get_samples()
        self.init_samples = init_samples

    def get_parameters(self, parameter_id):
        """
        获取TOP N 的邻近区间
        Parameters
        ----------
        parameter_id

        Returns
        -------

        """
        # 自动判断是第几轮
        _round = self.get_round(parameter_id)

        if _round == 0:
            return self.init_samples.iloc[parameter_id, :].to_dict()
        else:
            UtilSys.is_debug_mode() and log.info(f"\nStart at round {_round} with parameter_id = {parameter_id}")

            round_sample = self.get_round_constraint(_round, LHSSampling())

            sample_index = int(parameter_id - self.round_sample_size[_round - 1])
            UtilSys.is_debug_mode() and log.info(f"Sample index: {sample_index}")
            return round_sample.iloc[sample_index, :].to_dict()
