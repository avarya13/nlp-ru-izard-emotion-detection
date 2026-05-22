import pandas as pd
from sklearn.metrics import f1_score


def compute_f1_macro(y_true, y_pred):
    df = pd.DataFrame(
        [
            {"macro": f1_score(y_true[:, i], y_pred[:, i] > 0.5, average="macro")}
            for i in range(y_true.shape[1])
        ]
    )
    return df["macro"].mean()


def compute_f1_micro(y_true, y_pred):
    df = pd.DataFrame(
        [
            {"micro": f1_score(y_true[:, i], y_pred[:, i] > 0.5, average="micro")}
            for i in range(y_true.shape[1])
        ]
    )
    return df["micro"].mean()
