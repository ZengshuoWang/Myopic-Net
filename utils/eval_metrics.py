import torch


def _acc(preds, targets, num_k, verbose=0):
    assert (isinstance(preds, torch.Tensor) and
            isinstance(targets, torch.Tensor) and
            preds.is_cuda and targets.is_cuda)

    if verbose >= 2:
        print("calling acc...")

    assert (preds.shape == targets.shape)
    assert (preds.max() <= num_k and targets.max() <= num_k)

    acc = int((preds == targets).sum()) / float(preds.shape[0])

    return acc
