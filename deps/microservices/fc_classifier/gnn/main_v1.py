"""
python main.py --dataset_name D1 --model_name GCN  >> result.txt
python main.py --dataset_name D1 --model_name GAT  >> result.txt
python main.py --dataset_name D1 --model_name GraphSage  >> result.txt
python main.py --dataset_name D2 --model_name GCN  >> result.txt
python main.py --dataset_name D2 --model_name GAT  >> result.txt
python main.py --dataset_name D2 --model_name GraphSage >> result.txt
"""
import argparse
import sys
import os

from fcvgae.kvdb_json import KVDBJson
from microservices.fc_classifier.fcvgae.libs import *

sys.path.append(os.path.abspath("../../"))

import warnings

warnings.filterwarnings("ignore")

from gnn.gnn_lib import *
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument("-d", '--dataset_name', default="D1", type=str, help="dataset name, D1 or D2")
parser.add_argument("-m", "--model_name", type=str, default="GAT", help="model name, SVM or RF or ADABOOST")
if __name__ == '__main__':

    args = parser.parse_args()
    assert args.dataset_name in ["D1", "D2"]
    pl.seed_everything(42)
    if args.dataset_name == "D1":
        loader = D1()
    else:
        loader = D2()

    train_samples, test_samples = load_gnn_data(loader.DATA_NAME)

    gconf = GCNConf(
        n_node=train_samples[1][2].shape[0],
        n_fea=train_samples[1][2].shape[1],
        n_class=len(loader.TypeDict.keys()),
        lr=0.01
    )
    if args.model_name == "GAT":
        cla = GATClassifier(num_classes=gconf.n_class, in_channels=gconf.n_fea)
    elif args.model_name == "GCN":
        cla = GCNClassifier(num_classes=gconf.n_class, in_channels=gconf.n_fea)
    elif args.model_name == "GraphSage":
        cla = GraphSageClassifier(num_classes=gconf.n_class, in_channels=gconf.n_fea)
    else:
        raise ValueError("model_name must be RF or SVM or ADABOOST")

    gconf = GCNConf(
        n_node=train_samples[1][2].shape[0],
        n_fea=train_samples[1][2].shape[1],
        n_class=len(loader.TypeDict.keys()),
        lr=0.01
    )
    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=3, verbose=True, mode="min")

    print("train, test samples: ", len(train_samples), len(test_samples))
    train_loader = create_dataloader(train_samples, 128)
    test_loader = create_dataloader(test_samples, len(test_samples))

    trainer = pl.Trainer(
        max_epochs=1000,
        callbacks=[early_stop_callback],
        deterministic=True,
        fast_dev_run=True,
        enable_checkpointing=False)
    trainer.fit(model=cla, train_dataloaders=train_loader)
    predict = trainer.predict(model=cla, dataloaders=test_loader)

    # Extract and concatenate timestamps and predictions separately
    timestamps = np.concatenate([batch[0] for batch in predict], axis=0)
    predictions = np.concatenate([batch[1] for batch in predict], axis=0)

    # Combine into DataFrame
    predict_df = pd.DataFrame(np.column_stack((timestamps, predictions)), columns=['timestamp', 'predict'])

    prec, recall, f1 = eval_predict_failure_type_v2(loader, predict_df, desc=vars(args))
    fdb = KVDBJson()
    fdb.add(vars(args), {
        "prec": prec,
        "rec": recall,
        "f1": f1
    })
 