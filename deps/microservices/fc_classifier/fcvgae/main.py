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
import warnings

warnings.filterwarnings("ignore")
import sys
import torch
import pytorch_lightning
from fcvgae.mlp import MLPClassifier
from fcvgae.kvdb_json import KVDBJson

from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from fcvgae.fcvgae import FCVGAE
from fcvgae.libs import create_dataloader, D2, \
    PytorchLightingUtil, parse_args, get_reconstruction_error, get_class_weight, KVDBSqlite, D1

if __name__ == '__main__':
    db = KVDBJson()
    pytorch_lightning.seed_everything(42)
    args = parse_args()
    _key = vars(args)
    if db.query(_key) is not None:
        print("model has trained", {
            "hpy": _key,
            "f1": db.query(_key)["f1"]
        })
        sys.exit(0)

    print("args", args)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    if args.dataset.lower() == "d1":
        loader = D1()
    else:
        loader = D2()
    cases = loader.load_case()
    train_samples, test_samples = loader.load_samples()
    naive_model_path = loader.get_naive_model_path()
    vgae_batch_size = args.vgae_batch_size  # 32
    in_dim = loader.get_input_dim()
    vgae_hidden_dim = args.vgae_hidden_dim  # 64
    vgae_latent_dim = args.vgae_latent_dim
    vgae_drop_rate = args.vgae_drop_rate
    vgae_num_layers = args.vgae_num_layers
    vgae_model = FCVGAE(
        in_dim,
        hidden_feats=vgae_hidden_dim,
        latent_dim=vgae_latent_dim,
        out_feats=in_dim,
        dropout=vgae_drop_rate,
        num_layers=vgae_num_layers

    )
    device = "gpu" if torch.cuda.is_available() else "cpu"
    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0, patience=10, verbose=True, mode="min")
    dataloader = create_dataloader(train_samples, vgae_batch_size, shuffle=False)
    trainer = pytorch_lightning.Trainer(max_epochs=1000,
                                        fast_dev_run=args.debug == 1,
                                        callbacks=[early_stop_callback],
                                        deterministic=True,
                                        accelerator=device,
                                        enable_checkpointing=False,
                                        devices=[0] if device == "gpu" else None
                                        )
    trainer.fit(model=vgae_model, train_dataloaders=dataloader)

    rc_data = get_reconstruction_error(vgae_model, test_samples, cases, type_hash=loader.TypeHash,
                                       filter_ms_strategy=None)
    X = rc_data['fr']  # shape: (n_samples, n_features)
    y = rc_data['label']  # shape: (n_samples,)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.mlp_test_ratio,
        random_state=42,
        stratify=y
    )
    assert len(y_test) / len(y) == args.mlp_test_ratio
    batch_size = args.mlp_batch_size
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    input_dim = len(X_train[0])
    num_classes = loader.n_class
    class_weight = get_class_weight(y_test)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    model = MLPClassifier(input_dim, args.mlp_hidden_dims, num_classes, dropout=args.mlp_drop_rate,
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
        fast_dev_run=args.debug == 1,
        callbacks=[early_stop_callback],
        accelerator=device,
        deterministic=True,
        devices=[0] if device == "gpu" else None,
        enable_checkpointing=False
    )
    trainer.fit(model=model, train_dataloaders=train_loader)
    res = trainer.predict(model=model, dataloaders=test_loader)
    y_pred, y_true = PytorchLightingUtil.merage_predict_two(res)
    acc = round(precision_score(y_true, y_pred, average="weighted"), 4)
    recall = round(recall_score(y_true, y_pred, average="weighted"), 4)
    f1 = round(f1_score(y_true, y_pred, average="weighted"), 4)
    db.add(_key, {
        "f1": f1
    })

    print("==== Failrue Classification ====")
    print("precision:", acc)
    print("Recall (micro):", recall)
    print("F1 (micro):", f1)
    print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))
