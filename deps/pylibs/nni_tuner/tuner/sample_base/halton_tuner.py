from nni.common.hpo_utils import format_search_space

from pylibs.utils.util_log import get_logger
from tuner.sampling.halton_sampling_pysample import HaltonSamplingPySample
from tuner.base_tuner import BaseTuner

log = get_logger()


class HaltonTuner(BaseTuner):
    """
    A nni_tuner using Halton method. This nni_tuner only use the sample method to work.
    """

    def __init__(self, *args, **kwargs):
        """
        A sample_base nni_tuner

        Parameters
        ----------
        args :
        kwargs :
        """
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

        # metric = self.check_receive_metric(parameters, value)
        # self.trial_results.append({
        #     "parameter_id": parameter_id,
        #     "metric": metric,
        #     "parameters": parameters
        # })ddd
        UtilSys.is_debug_mode() and log.info(f"Receive trial:"
                                             f"\nparameter_id:{parameter_id},"
                                             f"\nparameters:{parameters},\n"
                                             f"\nvalue:{value},"
                                             f"**kwargs:{kwargs}")

    def generate_parameters(self, parameter_id, **kwargs):
        '''
        Returns a set of trial (hyper-)parameters, as a serializable object
        parameter_id: int

        '''
        # your code implements here.
        return self.generate_parameters_by_parameter_id(parameter_id)

    def update_search_space(self, search_space):
        '''
        Tuners are advised to support updating search space at run-time.
        If a nni_tuner can only set search space once before generating first hyper-parameters,
        it should explicitly document this behaviour.
        cur_search_space_: JSON object created by experiment owner, e.g.:
        {'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]}}
        '''
        # your code implements here.
        UtilSys.is_debug_mode() and log.info(f"update_search_space:\n{search_space}")
        self.format_search_space = format_search_space(search_space)
        self.search_space = search_space
        sampler = HaltonSamplingPySample(self.n_trails, self.search_space)
        init_samples = sampler.get_samples()
        self.init_samples = init_samples
