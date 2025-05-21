from unittest import TestCase

from numpy.testing import assert_almost_equal

from pylibs.common import HyperParameter


class TestHP(TestCase):
    def testCase(self):
        _hpy = {
            HyperParameter.BATCH_SIZE: 256,
            HyperParameter.HIDDEN_ACTIVATION: 'relu',
            HyperParameter.OUTPUT_ACTIVATION: 'relu',
            HyperParameter.L2_REGULARIZER: 0.1,
            HyperParameter.DROP_RATE: 0.2,
            HyperParameter.EPOCHS: 3,
            HyperParameter.LATENT_DIM: 1,
            HyperParameter.N_NEURONS_LAYER1: 5,
            HyperParameter.N_NEURONS_LAYER2: 2,
            HyperParameter.N_NEURONS_LAYER3: 3,
        }
        hpy=HyperParameter.valid_search_space(_hpy)
        assert_almost_equal([5,2,3],hpy[HyperParameter.ENCODER_NEURONS])
        assert_almost_equal([5,2,3],hpy[HyperParameter.DECODER_NEURONS][::-1])

    def testCase00(self):
        _hpy = {
            HyperParameter.BATCH_SIZE: 256,
            HyperParameter.HIDDEN_ACTIVATION: 'relu',
            HyperParameter.OUTPUT_ACTIVATION: 'relu',
            HyperParameter.L2_REGULARIZER: 0.1,
            HyperParameter.DROP_RATE: 0.2,
            HyperParameter.EPOCHS: 3,
            HyperParameter.LATENT_DIM: 1,
            HyperParameter.N_NEURONS_LAYER1: 5,
            HyperParameter.N_NEURONS_LAYER2: None,
            HyperParameter.N_NEURONS_LAYER3: None,
        }
        hpy=HyperParameter.valid_search_space(_hpy)
        assert_almost_equal([5],hpy[HyperParameter.ENCODER_NEURONS])
        assert_almost_equal([5],hpy[HyperParameter.DECODER_NEURONS][::-1])

    def testCase01(self):
        _hpy = {
            HyperParameter.BATCH_SIZE: 256,
            HyperParameter.HIDDEN_ACTIVATION: 'relu',
            HyperParameter.OUTPUT_ACTIVATION: 'relu',
            HyperParameter.L2_REGULARIZER: 0.1,
            HyperParameter.DROP_RATE: 0.2,
            HyperParameter.EPOCHS: 3,
            HyperParameter.LATENT_DIM: 1,
            HyperParameter.N_NEURONS_LAYER1: 5,
            HyperParameter.N_NEURONS_LAYER2: 3,
            HyperParameter.N_NEURONS_LAYER3: None,
        }
        hpy=HyperParameter.valid_search_space(_hpy)
        assert_almost_equal([5,3],hpy[HyperParameter.ENCODER_NEURONS])
        assert_almost_equal([5,3],hpy[HyperParameter.DECODER_NEURONS][::-1])


    def testCase2(self):
        _hpy = {
            HyperParameter.BATCH_SIZE: 256,
            HyperParameter.HIDDEN_ACTIVATION: 'relu',
            HyperParameter.OUTPUT_ACTIVATION: 'relu',
            HyperParameter.L2_REGULARIZER: 0.1,
            HyperParameter.DROP_RATE: 0.2,
            HyperParameter.EPOCHS: 3,
            HyperParameter.LATENT_DIM: 1,
            HyperParameter.N_NEURONS_LAYER1: 5,
            HyperParameter.N_NEURONS_LAYER2: 2,
        }
        target=[5,2]
        hpy=HyperParameter.valid_search_space(_hpy)
        assert_almost_equal(target,hpy[HyperParameter.ENCODER_NEURONS])
        assert_almost_equal(target,hpy[HyperParameter.DECODER_NEURONS][::-1])
    def testCase3(self):
        _hpy = {
            HyperParameter.BATCH_SIZE: 256,
            HyperParameter.HIDDEN_ACTIVATION: 'relu',
            HyperParameter.OUTPUT_ACTIVATION: 'relu',
            HyperParameter.L2_REGULARIZER: 0.1,
            HyperParameter.DROP_RATE: 0.2,
            HyperParameter.EPOCHS: 3,
            HyperParameter.LATENT_DIM: 1,
            HyperParameter.N_NEURONS_LAYER1: 5,
        }
        target=[5]
        hpy=HyperParameter.valid_search_space(_hpy)
        assert_almost_equal(target,hpy[HyperParameter.ENCODER_NEURONS])
        assert_almost_equal(target,hpy[HyperParameter.DECODER_NEURONS][::-1])


    def testCase4(self):
        _hpy = {
            HyperParameter.BATCH_SIZE: 256,
            HyperParameter.HIDDEN_ACTIVATION: 'relu',
            HyperParameter.OUTPUT_ACTIVATION: 'relu',
            HyperParameter.L2_REGULARIZER: 0.1,
            HyperParameter.DROP_RATE: 0.2,
            HyperParameter.EPOCHS: 3,
            HyperParameter.LATENT_DIM: 1,
            HyperParameter.N_NEURONS_LAYER3: 5,
        }
        target=[5]
        try:
            hpy=HyperParameter.valid_search_space(_hpy)
            assert False
        except Exception as e:
            assert str(e).find("n_neurons_layer2 and n_neurons_layer1 can't be None.") >-1
    def testCase5(self):
        _hpy = {
            HyperParameter.BATCH_SIZE: 256,
            HyperParameter.HIDDEN_ACTIVATION: 'relu',
            HyperParameter.OUTPUT_ACTIVATION: 'relu',
            HyperParameter.L2_REGULARIZER: 0.1,
            HyperParameter.DROP_RATE: 0.2,
            HyperParameter.EPOCHS: 3,
            HyperParameter.LATENT_DIM: 1,
            HyperParameter.N_NEURONS_LAYER2: 5,
        }
        target=[5]
        try:
            hpy=HyperParameter.valid_search_space(_hpy)
            assert False
        except Exception as e:
            assert str(e).find("n_neurons_layer1 can't be None.") >-1