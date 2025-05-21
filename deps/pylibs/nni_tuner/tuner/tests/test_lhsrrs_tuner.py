import numpy as np
from numpy.testing import assert_almost_equal
from unittest import TestCase

from tuner.mixture.lhsrrs_tuner import LHSRRSTuner


class TestLHSRun(TestCase):
    def test_trail1(self):
        # nni_tuner:
        #     user: LHSRNS
        #     class_args:
        #        optimize_mode: maximize
        #        n_trails: 3
        n_trails = 10
        round_rate = [0.7, 0.3]
        search_space = {
            "latent_dim": {
                "_type": "quniform",
                "_value": [
                    2,
                    20,
                    1
                ]
            },
            "activation_function": {
                "_type": "choice",
                "_value": [
                    "relu",
                    "softmax",
                    "tanh",
                    "leakyrelu"
                ]
            },
            "lr": {
                "_type": "uniform",
                "_value": [
                    1e-07,
                    0.01
                ]
            },
            "weight_decay": {
                "_type": "uniform",
                "_value": [
                    1e-07,
                    0.1
                ]
            },
            "window_size": {
                "_type": "quniform",
                "_value": [
                    16,
                    256,
                    1
                ]
            },
            "hidden_neurons": {
                "_type": "quniform",
                "_value": [
                    32,
                    256,
                    1
                ]
            },
            "num_l_samples": {
                "_type": "quniform",
                "_value": [
                    32,
                    512,
                    1
                ]
            },
            "batch_size": {
                "_type": "quniform",
                "_value": [
                    32,
                    256,
                    1
                ]
            },
            "step_size": {
                "_type": "quniform",
                "_value": [
                    1,
                    50,
                    1
                ]
            },
            "gamma": {
                "_type": "uniform",
                "_value": [
                    1e-07,
                    0.1
                ]
            },
            "epochs": {
                "_type": "quniform",
                "_value": [
                    50,
                    300,
                    1
                ]
            }
        }

        tuner = LHSRRSTuner(optimize_mode="maximize", n_trails=n_trails, round_rate=round_rate)
        tuner.update_search_space(search_space)

        init_sample_size = int((round_rate[0] * n_trails))
        print("\n")
        for i in range(n_trails):
            metric = np.random.random()
            if i < init_sample_size:
                parameters = tuner.generate_parameters(i)
                tuner.receive_trial_result(i, parameters, metric)
                print("init", metric, parameters)
            else:
                parameters = tuner.generate_parameters(i)
                tuner.receive_trial_result(i, parameters, metric)
                print("opt", metric, parameters)

    def test_trail2(self):
        # nni_tuner:
        #     user: LHSRNS
        #     class_args:
        #        optimize_mode: maximize
        #        n_trails: 3
        n_trails = 10
        round_rate = [0.7, 0.3]
        tuner = LHSRRSTuner(optimize_mode="maximize", n_trails=n_trails, round_rate=round_rate)
        tuner.update_search_space({'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]},
                                   'res': {'_type': 'quniform', '_value': [3, 8, 1]}})

        init_sample_size = int((round_rate[0] * n_trails))
        print("\n")
        for i in range(n_trails):
            metric = i
            if i < init_sample_size:
                parameters = tuner.generate_parameters(i)
                tuner.receive_trial_result(i, parameters, i)
                print("init", metric, parameters)
            else:
                parameters = tuner.generate_parameters(i)
                tuner.receive_trial_result(i, parameters, i)
                print("opt", metric, parameters)

    def test_trail3(self):
        # nni_tuner:
        #     user: LHSRNS
        #     class_args:
        #        optimize_mode: maximize
        #        n_trails: 3
        n_trails = 1
        round_rate = [1]
        tuner = LHSRRSTuner(optimize_mode="maximize", n_trails=n_trails, round_rate=round_rate)
        tuner.update_search_space({'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]},
                                   'res': {'_type': 'quniform', '_value': [3, 8, 1]}})

        init_sample_size = int((round_rate[0] * n_trails))
        print("\n")
        for i in range(n_trails):
            metric = i
            if i < init_sample_size:
                parameters = tuner.generate_parameters(i)
                tuner.receive_trial_result(i, parameters, i)
                print("init", metric, parameters)
            else:
                parameters = tuner.generate_parameters(i)
                tuner.receive_trial_result(i, parameters, i)
                print("opt", metric, parameters)

    def test_trail4(self):
        # nni_tuner:
        #     user: LHSRNS
        #     class_args:
        #        optimize_mode: maximize
        #        n_trails: 3
        n_trails = 2
        round_rate = [0.5, 0.5]
        tuner = LHSRRSTuner(optimize_mode="maximize", n_trails=n_trails, round_rate=round_rate)
        tuner.update_search_space({'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]},
                                   'res': {'_type': 'quniform', '_value': [3, 8, 1]}})

        print("\n")
        for parameter_id in range(n_trails):
            metric = parameter_id
            parameters = tuner.generate_parameters(parameter_id)
            tuner.receive_trial_result(parameter_id, parameters, metric)
            print(f"\n============trail {parameter_id}==============="
                  f"\nparameter_id: {parameter_id},"
                  f"\nmetric:{metric},"
                  f"\nparameters:{parameters}")

    def test_trail5(self):
        # nni_tuner:
        #     user: LHSRNS
        #     class_args:
        #        optimize_mode: maximize
        #        n_trails: 3
        n_trails = 3
        round_rate = [0.5, 0.5]
        tuner = LHSRRSTuner(optimize_mode="maximize", n_trails=n_trails, round_rate=round_rate)
        tuner.update_search_space({'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]},
                                   'res': {'_type': 'quniform', '_value': [3, 8, 1]}})

        print("\n")
        for parameter_id in range(n_trails):
            metric = parameter_id
            parameters = tuner.generate_parameters(parameter_id)
            tuner.receive_trial_result(parameter_id, parameters, metric)
            print(f"\n============trail {parameter_id}==============="
                  f"\nparameter_id: {parameter_id},"
                  f"\nmetric:{metric},"
                  f"\nparameters:{parameters}")

    def test_trail6(self):
        search_space = {
            'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]},
            'activation_function': {'_type': 'choice', '_value': ["relu", "softmax", "tanh", "leakyrelu"]},
            'lr': {'_type': 'uniform', '_value': [0.0000001, 0.1]},
            'weight_decay': {'_type': 'uniform', '_value': [0.0000001, 0.1]},
            'window_size': {'_type': 'quniform', '_value': [16, 256, 1]},
            'number_of_neural_per_layer': {'_type': 'quniform', '_value': [32, 256, 1]},
            'num_l_samples': {'_type': 'quniform', '_value': [32, 512, 1]},
            'batch_size': {'_type': 'quniform', '_value': [8, 256, 1]},
            'step_size': {'_type': 'quniform', '_value': [1, 50, 1]},
            'gamma': {'_type': 'uniform', '_value': [0.0000001, 0.1]},
            'n_epoch': {'_type': 'quniform', '_value': [50, 300, 1]},
        }
        n_trails = 3
        round_rate = [0.7, 0.3]
        tuner = LHSRRSTuner(optimize_mode="maximize", n_trails=n_trails, round_rate=round_rate)
        tuner.update_search_space(search_space)

        print("\n")
        for parameter_id in range(n_trails):
            metric = parameter_id
            parameters = tuner.generate_parameters(parameter_id)
            tuner.receive_trial_result(parameter_id, parameters, metric)
            print(f"\n============trail {parameter_id}==============="
                  f"\nparameter_id: {parameter_id},"
                  f"\nmetric:{metric},"
                  f"\nparameters:{parameters}")

    def test_trail7(self):
        search_space = {
            'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]},
            'activation_function': {'_type': 'choice', '_value': ["relu", "softmax", "tanh", "leakyrelu"]},
            'lr': {'_type': 'uniform', '_value': [0.0000001, 0.1]},
            'weight_decay': {'_type': 'uniform', '_value': [0.0000001, 0.1]},
            'window_size': {'_type': 'quniform', '_value': [16, 256, 1]},
            'number_of_neural_per_layer': {'_type': 'quniform', '_value': [32, 256, 1]},
            'num_l_samples': {'_type': 'quniform', '_value': [32, 512, 1]},
            'batch_size': {'_type': 'quniform', '_value': [8, 256, 1]},
            'step_size': {'_type': 'quniform', '_value': [1, 50, 1]},
            'gamma': {'_type': 'uniform', '_value': [0.0000001, 0.1]},
            'n_epoch': {'_type': 'quniform', '_value': [50, 300, 1]},
        }
        n_trails = 1
        round_rate = [0.7, 0.3]
        tuner = LHSRRSTuner(optimize_mode="maximize", n_trails=n_trails, round_rate=round_rate)
        tuner.update_search_space(search_space)

        assert_almost_equal(tuner.round_sample_size[-1], n_trails)
        print("\n")
        for parameter_id in range(n_trails):
            metric = parameter_id
            parameters = tuner.generate_parameters(parameter_id)
            tuner.receive_trial_result(parameter_id, parameters, metric)
            print(f"\n============trail {parameter_id}==============="
                  f"\nparameter_id: {parameter_id},"
                  f"\nmetric:{metric},"
                  f"\nparameters:{parameters}")

    def test_trail8(self):
        search_space = {
            'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]},
            'activation_function': {'_type': 'choice', '_value': ["relu", "softmax", "tanh", "leakyrelu"]},
            'lr': {'_type': 'uniform', '_value': [0.0000001, 0.1]},
            'weight_decay': {'_type': 'uniform', '_value': [0.0000001, 0.1]},
            'window_size': {'_type': 'quniform', '_value': [16, 256, 1]},
            'number_of_neural_per_layer': {'_type': 'quniform', '_value': [32, 256, 1]},
            'num_l_samples': {'_type': 'quniform', '_value': [32, 512, 1]},
            'batch_size': {'_type': 'quniform', '_value': [8, 256, 1]},
            'step_size': {'_type': 'quniform', '_value': [1, 50, 1]},
            'gamma': {'_type': 'uniform', '_value': [0.0000001, 0.1]},
            'n_epoch': {'_type': 'quniform', '_value': [50, 300, 1]},
        }
        n_trails = 30
        round_rate = [0.7, 0.3]
        tuner = LHSRRSTuner(optimize_mode="maximize", n_trails=n_trails, round_rate=round_rate)
        tuner.update_search_space(search_space)

        assert_almost_equal(tuner.round_sample_size[-1], n_trails)
        print("\n")
        for parameter_id in range(n_trails):
            metric = parameter_id
            parameters = tuner.generate_parameters(parameter_id)
            tuner.receive_trial_result(parameter_id, parameters, metric)
            print(f"\n============trail {parameter_id}==============="
                  f"\nparameter_id: {parameter_id},"
                  f"\nmetric:{metric},"
                  f"\nparameters:{parameters}")

    def test_trail9(self):
        search_space = {
            'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]},
            'activation_function': {'_type': 'choice', '_value': ["relu", "softmax", "tanh", "leakyrelu"]},
            'lr': {'_type': 'uniform', '_value': [0.0000001, 0.1]},
            'weight_decay': {'_type': 'uniform', '_value': [0.0000001, 0.1]},
            'window_size': {'_type': 'quniform', '_value': [16, 256, 1]},
            'number_of_neural_per_layer': {'_type': 'quniform', '_value': [32, 256, 1]},
            'num_l_samples': {'_type': 'quniform', '_value': [32, 512, 1]},
            'batch_size': {'_type': 'quniform', '_value': [8, 256, 1]},
            'step_size': {'_type': 'quniform', '_value': [1, 50, 1]},
            'gamma': {'_type': 'uniform', '_value': [0.0000001, 0.1]},
            'n_epoch': {'_type': 'quniform', '_value': [50, 300, 1]},
        }
        n_trails = 30
        round_rate = [0.7, 0.2, 0.1]
        tuner = LHSRRSTuner(optimize_mode="maximize", n_trails=n_trails, round_rate=round_rate)
        tuner.update_search_space(search_space)

        assert_almost_equal(tuner.round_sample_size[-1], n_trails)
        print("\n")
        for parameter_id in range(n_trails):
            metric = parameter_id
            parameters = tuner.generate_parameters(parameter_id)
            tuner.receive_trial_result(parameter_id, parameters, metric)
            print(f"\n============trail {parameter_id}==============="
                  f"\nparameter_id: {parameter_id},"
                  f"\nmetric:{metric},"
                  f"\nparameters:{parameters}")

    def test_trail10(self):
        search_space = {
            'latent_dim': {'_type': 'quniform', '_value': [2, 20, 1]},
            'activation_function': {'_type': 'choice', '_value': ["relu", "softmax", "tanh", "leakyrelu"]},
            'lr': {'_type': 'uniform', '_value': [0.0000001, 0.1]},
            'weight_decay': {'_type': 'uniform', '_value': [0.0000001, 0.1]},
            'window_size': {'_type': 'quniform', '_value': [16, 256, 1]},
            'number_of_neural_per_layer': {'_type': 'quniform', '_value': [32, 256, 1]},
            'num_l_samples': {'_type': 'quniform', '_value': [32, 512, 1]},
            'batch_size': {'_type': 'quniform', '_value': [8, 256, 1]},
            'step_size': {'_type': 'quniform', '_value': [1, 50, 1]},
            'gamma': {'_type': 'uniform', '_value': [0.0000001, 0.1]},
            'n_epoch': {'_type': 'quniform', '_value': [50, 300, 1]},
        }
        n_trails = 30
        round_rate = [0.7, 0.2, 0.1]
        tuner = LHSRRSTuner(optimize_mode="maximize", n_trails=n_trails, round_rate=round_rate)
        tuner.update_search_space(search_space)

        assert_almost_equal(tuner.round_sample_size[-1], n_trails)
        print("\n")
        for parameter_id in range(n_trails):
            metric = {
                "default": parameter_id,
                "value": np.random.random()
            }

            parameters = tuner.generate_parameters(parameter_id)
            tuner.receive_trial_result(parameter_id, parameters, metric)
            print(f"\n============trail {parameter_id}==============="
                  f"\nparameter_id: {parameter_id},"
                  f"\nmetric:{metric},"
                  f"\nparameters:{parameters}")

    def test_trail11(self):
        search_space = {
            "latent_dim": {
                "_type": "quniform",
                "_value": [
                    2,
                    20,
                    1
                ]
            },
            "hidden_activation": {
                "_type": "choice",
                "_value": [
                    "relu",
                    "sigmoid",
                    "softmax",
                    "softplus",
                    "softsign",
                    "tanh",
                    "selu",
                    "elu",
                    "exponential"
                ]
            },
            "output_activation": {
                "_type": "choice",
                "_value": [
                    "relu",
                    "sigmoid",
                    "softmax",
                    "softplus",
                    "softsign",
                    "tanh",
                    "selu",
                    "elu",
                    "exponential"
                ]
            },
            "lr": {
                "_type": "uniform",
                "_value": [
                    1e-07,
                    0.1
                ]
            },
            "lr_step": {
                "_type": "quniform",
                "_value": [
                    5,
                    50,
                    1
                ]
            },
            "lr_gamma": {
                "_type": "uniform",
                "_value": [
                    1e-07,
                    0.99
                ]
            },
            "l2_regularizer": {
                "_type": "uniform",
                "_value": [
                    1e-07,
                    0.1
                ]
            },
            "window_size": {
                "_type": "quniform",
                "_value": [
                    16,
                    256,
                    1
                ]
            },
            "hidden_layers": {
                "_type": "quniform",
                "_value": [
                    1,
                    5,
                    1
                ]
            },
            "hidden_neurons": {
                "_type": "quniform",
                "_value": [
                    32,
                    256,
                    1
                ]
            },
            "epochs": {
                "_type": "quniform",
                "_value": [
                    50,
                    300,
                    1
                ]
            },
            "batch_size": {
                "_type": "quniform",
                "_value": [
                    32,
                    256,
                    1
                ]
            },
            "dropout_rate": {
                "_type": "uniform",
                "_value": [
                    1e-07,
                    0.1
                ]
            }
        }

        n_trails = 10
        round_rate = [0.7, 0.3]
        tuner = LHSRRSTuner(optimize_mode="maximize", n_trails=n_trails, round_rate=round_rate)
        tuner.update_search_space(search_space)
        assert_almost_equal(tuner.round_sample_size[-1], n_trails)
        print("\n")
        for parameter_id in range(n_trails):
            metric = {
                "default": parameter_id,
                "value": np.random.random()
            }

            parameters = tuner.generate_parameters(parameter_id)
            tuner.receive_trial_result(parameter_id, parameters, metric)
            print(f"\n============trail {parameter_id}==============="
                  f"\nparameter_id: {parameter_id},"
                  f"\nmetric:{metric},"
                  f"\nparameters:{parameters}")
