import torch


def binary_accuracy(y_pred, y_true, avg=True, threshold=0.5):
    correct = (y_pred > threshold) == y_true

    return correct.sum() if avg is False else correct.type(torch.float32).mean()


def multiclass_accuracy(y_pred, y_true, avg=True):
    pred = y_pred.max(1, keepdim=True)[1]

    correct = pred.eq(y_true.view_as(pred))

    return correct.sum() if avg is False else correct.type(torch.float32).mean()
