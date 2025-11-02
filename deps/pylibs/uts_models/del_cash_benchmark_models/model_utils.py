

def create_dataset_for_cnn_and_lstm(X_clean):
    X_train, Y_train = X_clean[:, 0:-1], X_clean[:, -2:-1]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    return X_train, Y_train
