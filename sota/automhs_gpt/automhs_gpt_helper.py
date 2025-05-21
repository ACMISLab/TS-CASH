import time

import pandas as pd

from datasets.openml.meta_helper import MetaAutoCASH
from sota.auto_cash.auto_cash_helper import KVDB
from sota.automhs_gpt.util_open_ai import UtilOpenai
from tshpo.automl_libs import get_auto_sklearn_classification_search_space


class AutoMHS_GPT():
    def __init__(self, gpt_name="gpt-4o"):
        # cs = get_auto_sklearn_classification_search_space(y_train=[0, 1])
        self.gpt_name = gpt_name
        # self.cs_description = dict(cs)
        self.dataset_meta = pd.read_csv(
            "/Users/sunwu/SW-Research/AutoML-Benchmark/deps/datasets/openml/auto_cash_data_meta_fea.csv", index_col=0)

    def get_prompt_model_and_hpys(self, dataset):
        task_description = self.get_task_prompt(dataset)
        print(task_description)
        messages = [
            {"role": "user", "content": task_description}
        ]
        msg = UtilOpenai.chat_by_modelname(messages, self.gpt_name)
        parsed_msg = UtilOpenai.parse_message_without_stream(msg)
        print("✅ ", parsed_msg)
        return parsed_msg

    def get_dataset_meta(self, dataset):
        # ,id,dataid,data_name,mf0,mf2,mf4,mf6,mf7,mf9,mf13
        _dataset_meta = self.dataset_meta.query(f"data_name=='{dataset}'").iloc[0].to_dict()
        mac = MetaAutoCASH(**_dataset_meta)
        return mac.get_prompt_meta()

    def get_task_prompt(self, dataset):
        task_description = f"""
        I have a classification problem that uses the dataset {dataset}  and I want to know which is the best model with the defined hyperparameters for this task. The dataset is tabular and I only wish to receive a model and its hyperparameters. You must return only a JSON object including the keys 'model' and 'hyperparameters'.

        The search space is defined as: \n{self.get_cs_description()}\n
        The description of the dataset is: \n{self.get_dataset_meta(dataset)}
        """
        return task_description

    def get_cs_description(self):
        cs = get_auto_sklearn_classification_search_space(y_train=[0, 1])

        # print(cs.get_hyperparameter_names())
        # print(cs['__choice__'])
        prompt = ""
        for _model in cs['__choice__'].choices:
            print(_model)
            prompt += f"\nthe hyperparameters of {_model} are: "
            for _hyp_names in cs.get_hyperparameters():
                if _hyp_names.name.startswith(_model):
                    _hpy_name = _hyp_names.name.replace(f"{_model}:", "")
                    prompt += f"{_hpy_name},"
            prompt = prompt[:-1] + ";"

        return prompt


if __name__ == '__main__':
    datasets = ["dresses-sales",
                "climate-model-simulation-crashes",
                "cylinder-bands",
                "ilpd",
                "credit-approval",
                "breast-w",
                "diabetes",
                "tic-tac-toe",
                "credit-g",
                "qsar-biodeg",
                "pc1",
                "pc4",
                "pc3",
                "kc1",
                "ozone-level-8hr",
                "madelon",
                "kr-vs-kp",
                "Bioresponse",
                "sick",
                "spambase",
                "wilt",
                "churn",
                "phoneme",
                "jm1",
                "PhishingWebsites",
                "nomao",
                "bank-marketing",
                "electricity",
                ]
    DEBUG = True
    kvdb = KVDB("authmhs_gpt.dump", "./")
    for _gpt_name in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]:
        for _dataset in datasets:

            _dbkey = f"{_dataset}:::{_gpt_name}"
            if kvdb.query(_dbkey, is_return_ori=True) is None:
                amgpt = AutoMHS_GPT(gpt_name=_gpt_name)
                suggest = amgpt.get_prompt_model_and_hpys(_dataset)
                kvdb.add(_dbkey, {
                    "prompt": amgpt.get_task_prompt(_dataset),
                    "suggest": suggest
                })
            else:
                print("cash is exists")
    print("✅✅✅✅✅")
