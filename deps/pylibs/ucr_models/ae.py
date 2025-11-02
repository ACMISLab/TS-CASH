#  pyod-2.0.1
import numpy as np
import torch
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import mean_absolute_error
from datasets.ucr.ucr_dataset_loader import calc_acc_score


class AE(AutoEncoder):
    def accuracy_score(self, X_test, anomaly_range):
        # 确保 X_test 是一个 PyTorch 张量
        X_test = torch.from_numpy(X_test).float()
        if torch.cuda.is_available():
            X_test = X_test.to('cuda')

        X_pred = self.model.forward(X_test)
        X_pred = X_pred.cpu().detach().numpy()

        X_pred = np.nan_to_num(X_pred)
        # print("++++++++++++++result.shape:", X_pred.shape)

        X_score = []
        for test, pred in zip(X_test, X_pred):
            score = mean_absolute_error(test, pred)
            X_score.append(score)

        self.anomaly_scores_ = X_score
        self.anomaly_pos_ = np.argmax(X_score)
        score = calc_acc_score(anomaly_range, self.anomaly_pos_, len(X_test))
        return score


if __name__ == '__main__':
    from libs import eval_model
    vae = AE(epoch_num=30)
    eval_model(vae,False)
    # HOME = os.path.join("figs", Path(sys.argv[0]).stem)
    # os.makedirs(HOME, exist_ok=True)
    # from ucr.ucr_dataset_loader import calc_acc_score, load_ucr_by_number
    # # 展示test
    # import matplotlib.pyplot as plt
    #
    # WINDOW_SIZE = 128
    # for i in range(1, 250):
    #     X_train_window, x_test_window, anomaly_range = load_ucr_by_number(i, window_length=WINDOW_SIZE)
    #     vae = AE(epoch_num=30)
    #     vae.fit(X_train_window)
    #     acc_score = vae.accuracy_score(x_test_window, anomaly_range)
    #     score = vae.score_
    #     detect_pose = vae.detect_pos_
    #
    #     fig, axes = plt.subplots(2, 1, figsize=(12, 4))
    #     fig.subplots_adjust(hspace=0.5)  # 0.5 是子图之间的垂直间距，可以根据需要调整
    #     axes[0].set_title("Red span: ground truth anomaly")
    #     ax = axes[0]
    #     ax.axvspan(anomaly_range[0], anomaly_range[1], color='red', alpha=0.8, label='红色区域')
    #     ax.plot(x_test_window[:, -1])
    #
    #     acc = calc_acc_score(anomaly_range, np.argmax(score), len(x_test_window))
    #     axes[1].set_title(f"Vertical line: anomaly with maximum score, acc score: {acc}")
    #     axes[1].plot(score)
    #     axes[1].axvline(x=np.argmax(score), color='red', linestyle='--', label='垂直线 x=5')
    #
    #     fig.savefig(f"{HOME}/data_{i}.png")
    #     # break
