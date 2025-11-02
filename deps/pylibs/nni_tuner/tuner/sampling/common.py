class TestCommon:
    search_space_01 = {
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
