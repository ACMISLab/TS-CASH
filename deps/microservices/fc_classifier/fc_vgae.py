"""
CPU环境下的依赖:
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
pip install dgl==0.9.0 -f https://data.dgl.ai/wheels/repo.html
pip install numpy==1.24
pip install "pip<24.1"
pip install pytorch-lightning==1.7.7
pip install "torchmetrics==0.10.3"

GPU
pip install torch==1.12.1+cu113 \
            torchvision==0.13.1+cu113 \
            torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install dgl-cu113==0.9.0 -f https://data.dgl.ai/wheels/repo.html
pip install numpy==1.24

"""
import sys

from pyutils.util_pytorch_lighting import get_trainer

sys.path.append("/Users/xxx/Research/dev_libs")
sys.path.append("/Volumes/sw_data/phd_paper_data/experiments/ts_cash_ms")
sys.path.append("/remote-home4/cs_acmis_xxx/tshpo_ms")
import copy
import warnings
from typing import Optional

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from pytorch_lightning import seed_everything
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA
from autosklearn.pipeline.implementations.util import (
    convert_multioutput_multiclass_to_multilabel,
)
from pyutils.kvdb.kvdb_json import KVDBJson

warnings.filterwarnings("ignore")
import torch
import pytorch_lightning
from fcvgae.mlp import MLPClassifier
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from fcvgae.fcvgae import FCVGAE
from fcvgae.libs import create_dataloader, D2, \
    get_reconstruction_error, get_class_weight, D1, PytorchLightingUtil

from ms_libs import get_device


class FCVGAEClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(
            self,
            vgae_batch_size,
            vgae_hidden_dim,
            vgae_latent_dim,
            vgae_drop_rate,
            vgae_num_layers,
            mlp_hidden_dims,
            mlp_batch_size,
            mlp_drop_rate,
            dataset_name="d1",
            mlp_test_ratio=0.4,
            random_state=None,
            debug=True,
            train_ratio=1
    ):
        self.vgae_batch_size = vgae_batch_size
        self.vgae_hidden_dim = vgae_hidden_dim
        self.vgae_latent_dim = vgae_latent_dim
        self.vgae_drop_rate = vgae_drop_rate
        self.vgae_num_layers = vgae_num_layers
        self.mlp_hidden_dims = [mlp_hidden_dims, mlp_hidden_dims // 2]
        self.mlp_batch_size = mlp_batch_size
        self.mlp_drop_rate = mlp_drop_rate
        self.mlp_test_ratio = mlp_test_ratio
        self.dataset = dataset_name
        self.random_state = random_state
        self.vgae_model = None
        self.mlp_model = None
        self.loader = None
        self.debug = debug
        self.train_ratio = train_ratio

    def fit(self, X=None, y=None, sample_weight=None):
        pytorch_lightning.seed_everything(42 if self.random_state is None else self.random_state)
        device = get_device()
        # Initialize data loader
        if self.dataset.lower() == "d1":
            self.loader = D1()
        else:
            self.loader = D2()

        cases = self.loader.load_case()
        self.train_samples, self.test_samples = self.loader.load_samples()
        # 在train_samples 中随机抽取train_ratio 比例的数据
        if self.train_ratio < 1:
            # 随机从训练数据集中抽取 self.train_ratio 比例的数据
            _samples = copy.deepcopy(self.train_samples)
            np.random.shuffle(_samples)
            self.train_samples = _samples[:int(len(_samples) * self.train_ratio)]

        # Convert hyperparameters to appropriate types
        vgae_batch_size = int(self.vgae_batch_size)
        vgae_hidden_dim = int(self.vgae_hidden_dim)
        vgae_latent_dim = int(self.vgae_latent_dim)
        vgae_drop_rate = float(self.vgae_drop_rate)
        vgae_num_layers = int(self.vgae_num_layers)

        in_dim = self.loader.get_input_dim()

        # Train VGAE model
        self.vgae_model = FCVGAE(
            in_dim,
            hidden_feats=vgae_hidden_dim,
            latent_dim=vgae_latent_dim,
            out_feats=in_dim,
            dropout=vgae_drop_rate,
            num_layers=vgae_num_layers
        )

        train_dataloader = create_dataloader(self.train_samples, vgae_batch_size, shuffle=False)

        trainer = get_trainer(debug=self.debug)
        trainer.fit(model=self.vgae_model, train_dataloaders=train_dataloader)
        rc_data = get_reconstruction_error(self.vgae_model, self.test_samples, cases, type_hash=self.loader.TypeHash,
                                           filter_ms_strategy=None)
        X = rc_data['fr']  # shape: (n_samples, n_features)
        y = rc_data['label']  # shape: (n_samples,)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.mlp_test_ratio,
            random_state=self.random_state,
            stratify=y
        )
        batch_size = self.mlp_batch_size
        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        input_dim = len(X_train[0])
        num_classes = self.loader.n_class
        class_weight = get_class_weight(y_test)
        model = MLPClassifier(input_dim, self.mlp_hidden_dims, num_classes, dropout=self.mlp_drop_rate,
                              class_weight=class_weight)

        early_stop_callback = EarlyStopping(
            monitor="train_loss",
            patience=50,
            min_delta=1e-4,
            verbose=True
        )
        trainer = pytorch_lightning.Trainer(
            max_epochs=1000,
            min_epochs=200,
            fast_dev_run=self.debug,
            callbacks=[early_stop_callback],
            accelerator=device,
            deterministic=True,
            devices=[0] if device == "gpu" else None,
            enable_checkpointing=False
        )
        trainer.fit(model=model, train_dataloaders=train_loader)
        res = trainer.predict(model=model, dataloaders=test_loader)
        y_pred, y_true = PytorchLightingUtil.merage_predict_two(res)
        prec = round(precision_score(y_true, y_pred, average="weighted"), 4)
        recall = round(recall_score(y_true, y_pred, average="weighted"), 4)
        f1 = round(f1_score(y_true, y_pred, average="weighted"), 4)

        print("==== Failrue Classification ====")
        print("precision:", prec)
        print("Recall (micro):", recall)
        print("F1 (micro):", f1)
        print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))
        return prec, recall, f1

    def predict(self, X):
        if self.vgae_model is None or self.mlp_model is None:
            raise NotImplementedError("Model must be fitted before prediction")

        # For prediction, we need to process X through the VGAE first
        # This is a simplified version - in practice, you might need to adapt this
        # based on how your data flows through the VGAE model

        # Convert to tensor and get features through MLP
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.mlp_model.eval()
        with torch.no_grad():
            predictions = self.mlp_model(X_tensor)
            predictions = torch.argmax(predictions, dim=1)

        return predictions.numpy()

    def predict_proba(self, X):
        if self.vgae_model is None or self.mlp_model is None:
            raise NotImplementedError("Model must be fitted before prediction")

        # Convert to tensor and get probabilities through MLP
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.mlp_model.eval()
        with torch.no_grad():
            probas = torch.softmax(self.mlp_model(X_tensor), dim=1)

        probas = probas.numpy()
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "FCVGAE",
            "name": "FC-VGAE Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": False,
            "is_deterministic": False,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
            feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None, seed=42
    ):
        cs = ConfigurationSpace(seed=seed)

        vgae_batch_size = UniformIntegerHyperparameter(
            "vgae_batch_size", 64, 512, default_value=64
        )
        vgae_hidden_dim = UniformIntegerHyperparameter(
            "vgae_hidden_dim", 8, 256, default_value=8
        )
        vgae_latent_dim = UniformIntegerHyperparameter(
            "vgae_latent_dim", 16, 128, default_value=16
        )
        vgae_drop_rate = UniformFloatHyperparameter(
            "vgae_drop_rate", 0.0, 0.5, default_value=0.1
        )
        vgae_num_layers = UniformIntegerHyperparameter(
            "vgae_num_layers", 2, 10, default_value=5
        )
        mlp_hidden_dims = UniformIntegerHyperparameter(
            "mlp_hidden_dims", 64, 512, default_value=128,
        )
        mlp_batch_size = UniformIntegerHyperparameter(
            "mlp_batch_size", 16, 128, default_value=32
        )
        mlp_drop_rate = UniformFloatHyperparameter(
            "mlp_drop_rate", 0.1, 0.5, default_value=0.1
        )

        cs.add_hyperparameters([
            vgae_batch_size,
            vgae_hidden_dim,
            vgae_latent_dim,
            vgae_drop_rate,
            vgae_num_layers,
            mlp_hidden_dims,
            mlp_batch_size,
            mlp_drop_rate,
        ])

        return cs


if __name__ == '__main__':
    seed_everything(42)
    cs = FCVGAEClassifier.get_hyperparameter_search_space()

    configs = cs.sample_configuration(50)
    db = KVDBJson("fc_vgae.json")
    train_ratio = 0.1
    debug = True
    for dataset_name in ["D1", "D2"]:
        for config in configs:
            keys = config.get_dictionary()
            keys["dataset_name"] = dataset_name
            keys["train_ratio"] = train_ratio
            if db.is_exist(keys):
                continue
            # 创建分类器实例
            classifier = FCVGAEClassifier(
                debug=debug,
                dataset_name=dataset_name,
                train_ratio=train_ratio,
                **config
            )

            pre, recall, f1 = classifier.fit(X=None, y=None)

            db.add(keys, {"pre": pre, "recall": recall, "f1": f1})
