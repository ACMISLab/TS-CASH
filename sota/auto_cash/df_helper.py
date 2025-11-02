import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categorical_columns(df):
    # 创建一个LabelEncoder对象
    label_encoders = {}

    # 遍历DataFrame中的列
    for column in df.columns:
        # 如果列的数据类型是object，说明可能是分类变量
        if df[column].dtype == 'object':
            # 创建LabelEncoder并转换数据
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    return df, label_encoders


class DFHelper:
    @staticmethod
    def pre_process_df(df):
        """
        输入: df, n-1列: 特征; -1列: label
        1. 将cat列转为字符串
        2. 完成归一化
        3. 用均值填充nan
        """
        assert "label" in df.columns, "df must have a 'label' column"

        # 调用函数进行编码
        df_encoded, encoders = encode_categorical_columns(df)

        # 打印结果
        x_train = df_encoded.iloc[:, :-1]
        y_train = df_encoded.iloc[:, -1]
        mean = x_train.mean()
        std = x_train.std()
        normalize_train_x = (x_train - mean) / (std + 1e-9)
        normalize_train_x = normalize_train_x.fillna(mean)
        y_train = y_train.fillna(0)  # 用 0 填充 autocash 的  fscore
        return normalize_train_x, y_train


if __name__ == '__main__':
    pass
