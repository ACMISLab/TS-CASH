import os.path

PROJECT_HOME = os.environ["TSHPO_HOME"]


class TSHPOCommon:
    all_select_datasets = ["dresses-sales",
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
    ETCD_SERVER = "etcd.gwusun.top"

    @staticmethod
    def get_result_data(file_name):
        return os.path.join(PROJECT_HOME, "exp_results/tshpo", file_name)
