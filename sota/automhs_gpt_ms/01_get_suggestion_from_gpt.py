from fcvgae.kvdb_json import KVDBJson
from sota.auto_cash.auto_cash_helper import KVDB
from sota.automhs_gpt.automhs_gpt_helper import AutoMHS_GPT, AutoMHS_GPT_MS

datasets = ["D1",
            "D2",
            ]
DEBUG = True
kvdb = KVDBJson("authmhs_gpt_ms.json")
for _gpt_name in ["gpt-4o"]:
    for _dataset in datasets:
        _dbkey = {
            "dataset": _dataset,
            "gpt_name": _gpt_name
        }
        if kvdb.query(_dbkey) is None:
            amgpt = AutoMHS_GPT_MS(gpt_name=_gpt_name)
            suggest = amgpt.get_prompt_model_and_hpys(_dataset)
            kvdb.add(_dbkey, {
                "prompt": amgpt.get_task_prompt(_dataset),
                "suggest": suggest
            })
        else:
            print("cash is exists")
print("✅✅✅✅✅")
