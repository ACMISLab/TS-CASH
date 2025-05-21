from pylibs.uts_models.benchmark_models.tsbuad.models.cnn import cnn
from pylibs.experiments.example_helper import ExampleHelper

model = cnn(slidingwindow=ExampleHelper.WINDOW_SIZE, epochs=ExampleHelper.EPOCH, batch_size=ExampleHelper.BATCH_SIZE)
ExampleHelper.observation_model(model)
