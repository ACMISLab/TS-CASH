pip install pysampling nni==2.10
cd "$( cd "$( dirname "$0" )" >/dev/null 2>&1 && pwd )"
#pwd
python setup.py develop
nnictl algo register -m tuner/nni_register_config/nni_sample_base_config/dds_tuner.yaml
nnictl algo register -m tuner/nni_register_config/nni_sample_base_config/halton_tuner.yaml
nnictl algo register -m tuner/nni_register_config/nni_sample_base_config/lhs_tuner.yaml
nnictl algo register -m tuner/nni_register_config/nni_sample_base_config/rs_tuner.yaml
nnictl algo register -m tuner/nni_register_config/nni_sample_base_config/scbol_tuner.yaml
nnictl algo list
