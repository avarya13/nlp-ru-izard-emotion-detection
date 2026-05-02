from sklearn.metrics import roc_auc_score, f1_score


def compute_auc(y_true, y_pred):
    """y_true, y_pred: numpy arrays of shape (n_samples, n_labels)"""
    return roc_auc_score(y_true, y_pred, average=None)


def compute_f1_macro(y_true, y_pred):
    # threshold 0.5
    y_pred_bin = (y_pred > 0.5).astype(int)
    return f1_score(y_true, y_pred_bin, average="macro")
