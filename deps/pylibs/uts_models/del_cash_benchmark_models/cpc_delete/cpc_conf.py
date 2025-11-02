from timeseries_models.base_conf import BaseModelConfig


class CPCConf(BaseModelConfig):
    def update_parameters(self, parameters: dict):
        pass

    def __init__(self):
        super().__init__()

        # custom parameters
        self.epochs = 150
        self.n_warmup_steps = 100
        self.batch_size = 256
        self.sequence_length = 16
        self.timestep = 1
        self.masked_frames = 0
        self.cuda = None
        self.seed = 1
        self.log_interval = 50
        self.input_channel = 1
        self.hidden_size = 64
        self.time_step = 2
