from torchmetrics import F1Score, Accuracy
from sklearn.metrics import f1_score, accuracy_score
from configs import filter_config

CFG = filter_config.get_config()


def get_metrics(y_pred, y_true):
    # acc = Accuracy(num_classes=CFG.num_classes,).to(
    #     CFG.device
    # )(y_pred, y_true)
    # f1_score = F1Score(num_classes=CFG.num_classes,).to(
    #     CFG.device
    # )(y_pred, y_true)
    # y_true, y_pred = y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return acc, f1
