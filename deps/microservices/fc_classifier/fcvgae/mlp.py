from typing import Any
import pytorch_lightning
import torch
import torch.nn as nn

class MLPClassifier(pytorch_lightning.LightningModule):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.1, class_weight=None):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
        self.class_weight = class_weight
        if self.class_weight is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weight)

    def training_step(self, batch, batch_index):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def forward(self, x):
        return self.mlp(x)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        logits = self(x)
        predict_class = torch.argmax(logits, dim=1)
        return (predict_class, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer