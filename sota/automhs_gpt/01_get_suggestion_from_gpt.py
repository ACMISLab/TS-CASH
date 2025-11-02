from sota.auto_cash.auto_cash_helper import KVDB
from sota.automhs_gpt.automhs_gpt_helper import AutoMHS_GPT

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
        if kvdb.query(_dbkey) is None:
            amgpt = AutoMHS_GPT(gpt_name=_gpt_name)
            suggest = amgpt.get_prompt_model_and_hpys(_dataset)
            kvdb.add(_dbkey, {
                "prompt": amgpt.get_task_prompt(_dataset),
                "suggest": suggest
            })
        else:
            print("cash is exists")
print("✅✅✅✅✅")
