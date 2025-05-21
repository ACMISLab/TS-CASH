import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preproc_data(data, features):
    '''
    Simple preproc data:* LabelEncoded target
                        * One Hot Encoding cat_features
                        * Clean Nans as .median()
                        * split data on X, y

    data: pd.DataFrame()
    cat_features: list() # categorical variables in df

    data: DataFrame, 包含Feature和target
    features: 每一列的属性，比如
    'foreign_worker' = {str} 'categorical'
    'class' = {str} 'target'
    '''

    cat_features = []
    numeric_features = []
    target_col = None
    for fea_name, ftype in features.items():
        if ftype == "categorical":
            cat_features.append(fea_name)
        elif ftype == "numerical":
            numeric_features.append(fea_name)
        elif ftype == 'target':
            target_col = fea_name
            pass
        else:
            raise RuntimeError(f"Unsupported type {ftype}")

    # LabelEncoded Target
    data[target_col] = data[target_col].astype('category').cat.codes

    assert target_col is not None, f"Error for loading dataset {data},please checking original dataset."
    y = data[target_col]
    X = data.drop([target_col], axis=1)

    # Label Encoded Binary Features
    print("Label encoding...")
    for feature in X.columns:
        if (X[feature].nunique(dropna=False) < 3):
            X[feature] = X[feature].astype('category').cat.codes
            if len(cat_features) > 0:
                if feature in cat_features:
                    cat_features.remove(feature)

    # One Hot Encoding
    # if len(cat_features) > 0:
    #     encoder = OneHotEncoder(cols=cat_features, drop_invariant=True)
    #     X = encoder.fit_transform(X)
    # One Hot Encoding using Pandas
    # 假设 X 和 cat_features 已经定义
    print("Label encoding...")

    if len(cat_features) > 0:
        for feature in cat_features:
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature])

    # Standardize Numeric Features
    if len(numeric_features) > 0:
        scaler = StandardScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # Nans
    nan_columns = list(X.columns[X.isnull().sum() > 0])
    if nan_columns:
        for nan_column in nan_columns:
            X[nan_column + 'isNAN'] = pd.isna(X[nan_column]).astype('uint8')
        X[nan_columns].fillna(X[nan_columns].median(), inplace=True)

    # 确保均值约等于0, 方差约等于1
    # assert X[numeric_features].mean(axis=0).sum() < 0.0001, f"std is not equal to 0 for {data} "
    # assert round(X[numeric_features].std().mean(), 2) == 1.0, f"mean is not equal to 1 for {data} "
    return (X, y)
