import torch
import sys
from datetime import datetime

from .transforms import sobel_process, gray_process
from .eval_metrics import _acc


def _classify_get_data(config, net, dataloader, sobel, gray, get_soft=False, verbose=0):
    """
       Returns cuda tensors for flat preds and targets.
    """
    num_batches = len(dataloader)
    flat_targets_all = torch.zeros((num_batches * config.batch_sz), dtype=torch.int32).cuda()
    flat_predss_all = torch.zeros((num_batches * config.batch_sz), dtype=torch.int32).cuda()

    if get_soft:
        soft_predss_all = torch.zeros((num_batches * config.batch_sz, config.output_k),
                                      dtype=torch.float32).cuda()

    num_test = 0
    for b_i, batch in enumerate(dataloader):
        imgs_1 = batch[0].cuda()
        imgs_2 = batch[1].cuda()

        if sobel:
            imgs_1 = sobel_process(imgs_1, config.include_rgb)
            imgs_2 = sobel_process(imgs_2, config.include_rgb)

        if gray:
            imgs_1 = gray_process(imgs_1)
            imgs_2 = gray_process(imgs_2)

        flat_targets = batch[2]

        with torch.no_grad():
            x_outs = net(imgs_1, imgs_2)

        assert (x_outs.shape[1] == config.output_k)
        assert (len(x_outs.shape) == 2)

        num_test_curr = flat_targets.shape[0]
        num_test += num_test_curr

        start_i = b_i * config.batch_sz
        x_outs_curr = x_outs
        flat_preds_curr = torch.argmax(x_outs_curr, dim=1)  # along output_k
        flat_predss_all[start_i:(start_i + num_test_curr)] = flat_preds_curr

        if get_soft:
            soft_predss_all[start_i:(start_i + num_test_curr), :] = x_outs_curr

        flat_targets_all[start_i:(start_i + num_test_curr)] = flat_targets.cuda()

    flat_predss_all = flat_predss_all[:num_test]
    flat_targets_all = flat_targets_all[:num_test]

    if not get_soft:
        return flat_predss_all, flat_targets_all
    else:
        soft_predss_all = soft_predss_all[:num_test]

        return flat_predss_all, flat_targets_all, soft_predss_all


def _classify_get_data_BCEloss(config, net, dataloader, sobel, gray, get_soft=False, verbose=0):
    """
       Returns cuda tensors for flat preds and targets.
    """
    num_batches = len(dataloader)
    flat_targets_all = torch.zeros((num_batches * config.batch_sz), dtype=torch.int32).cuda()
    flat_predss_all = torch.zeros((num_batches * config.batch_sz), dtype=torch.int32).cuda()

    if get_soft:
        soft_predss_all = torch.zeros((num_batches * config.batch_sz, config.output_k),
                                      dtype=torch.float32).cuda()

    num_test = 0
    sigmoid = torch.nn.Sigmoid()
    for b_i, batch in enumerate(dataloader):
        imgs_1 = batch[0].cuda()
        imgs_2 = batch[1].cuda()

        if sobel:
            imgs_1 = sobel_process(imgs_1, config.include_rgb)
            imgs_2 = sobel_process(imgs_2, config.include_rgb)

        if gray:
            imgs_1 = gray_process(imgs_1)
            imgs_2 = gray_process(imgs_2)

        flat_targets = batch[2]

        with torch.no_grad():
            x_outs = net(imgs_1, imgs_2)
            x_outs = sigmoid(x_outs)

        assert (x_outs.shape[1] == config.output_k)
        assert (len(x_outs.shape) == 2)

        num_test_curr = flat_targets.shape[0]
        num_test += num_test_curr

        start_i = b_i * config.batch_sz
        x_outs_curr = x_outs.squeeze(dim=-1)
        flat_preds_curr = torch.where(x_outs_curr > 0.5, torch.ones_like(x_outs_curr, dtype=torch.int32),
                                      torch.zeros_like(x_outs_curr, dtype=torch.int32))
        flat_predss_all[start_i:(start_i + num_test_curr)] = flat_preds_curr

        if get_soft:
            soft_predss_all[start_i:(start_i + num_test_curr), :] = x_outs_curr

        flat_targets_all[start_i:(start_i + num_test_curr)] = flat_targets.cuda()

    flat_predss_all = flat_predss_all[:num_test]
    flat_targets_all = flat_targets_all[:num_test]

    if not get_soft:
        return flat_predss_all, flat_targets_all
    else:
        soft_predss_all = soft_predss_all[:num_test]

        return flat_predss_all, flat_targets_all, soft_predss_all


def _get_eval_data_acc(net, dataloader_eval, config, sobel, gray, get_data_fn=None, verbose=0):
    if verbose:
        print("calling classify eval direct (helper) %s" % datetime.now())
        sys.stdout.flush()

    flat_predss_all, flat_targets_all = get_data_fn(config, net, dataloader_eval,
                                                    sobel=sobel, gray=gray, verbose=verbose)

    if verbose:
        print("getting data fn has completed %s" % datetime.now())
        print("flat_targets_all %s, flat_predss_all %s" % (list(flat_targets_all.shape),
                                                           list(flat_predss_all.shape)))
        sys.stdout.flush()

    num_test = flat_targets_all.shape[0]
    if verbose == 2:
        print("num_test: %d" % num_test)
        for c in range(config.gt_k):
            print("gt_k: %d count: %d" % (c, (flat_targets_all == c).sum()))

    assert (flat_predss_all.shape == flat_targets_all.shape)

    if verbose:
        print("starting with eval mode %s, %s" % (config.eval_mode, datetime.now()))
        sys.stdout.flush()

    acc = _acc(flat_predss_all, flat_targets_all, config.gt_k, verbose)

    return acc


def classify_head_eval(config, net, dataloader_eval, dataloader_train_eval,
                       sobel, gray, get_data_fn=_classify_get_data, verbose=0):
    eval_acc = _get_eval_data_acc(net, dataloader_eval, config,
                                  sobel=sobel, gray=gray, get_data_fn=get_data_fn, verbose=verbose)
    train_eval_acc = _get_eval_data_acc(net, dataloader_train_eval, config,
                                        sobel=sobel, gray=gray, get_data_fn=get_data_fn, verbose=verbose)

    return {
        "eval_acc": eval_acc,
        "train_eval_acc": train_eval_acc
    }


def classify_head_eval_BCEloss(config, net, dataloader_eval, dataloader_train_eval,
                               sobel, gray, get_data_fn=_classify_get_data_BCEloss, verbose=0):
    eval_acc = _get_eval_data_acc(net, dataloader_eval, config,
                                  sobel=sobel, gray=gray, get_data_fn=get_data_fn, verbose=verbose)
    train_eval_acc = _get_eval_data_acc(net, dataloader_train_eval, config,
                                        sobel=sobel, gray=gray, get_data_fn=get_data_fn, verbose=verbose)

    return {
        "eval_acc": eval_acc,
        "train_eval_acc": train_eval_acc
    }


def classify_eval(config, net, dataloader_eval, dataloader_train_eval, sobel, gray, print_stats=False):
    if config.double_eval:
        stats_dict2 = classify_head_eval(config, net, dataloader_eval=dataloader_eval,
                                         dataloader_train_eval=dataloader_train_eval,
                                         sobel=sobel, gray=gray)

        if print_stats:
            print("double eval stats:")
            print(stats_dict2)
        else:
            config.double_eval_stats.append(stats_dict2)
            config.double_eval_acc.append(stats_dict2["eval_acc"])
            config.double_train_acc.append(stats_dict2["train_eval_acc"])

    net.eval()
    stats_dict = classify_head_eval(config, net, dataloader_eval=dataloader_eval,
                                    dataloader_train_eval=dataloader_train_eval,
                                    sobel=sobel, gray=gray)
    net.train()

    if print_stats:
        print("eval stats:")
        print(stats_dict)
    else:
        acc = stats_dict["eval_acc"]
        train_acc = stats_dict["train_eval_acc"]
        is_best = (len(config.epoch_acc) > 0) and (acc > max(config.epoch_acc))

        config.epoch_stats.append(stats_dict)
        config.epoch_acc.append(acc)
        config.epoch_train_acc.append(train_acc)

        return is_best


def classify_eval_BCEloss(config, net, dataloader_eval, dataloader_train_eval, sobel, gray, print_stats=False):
    if config.double_eval:
        stats_dict2 = classify_head_eval_BCEloss(config, net, dataloader_eval=dataloader_eval,
                                                 dataloader_train_eval=dataloader_train_eval,
                                                 sobel=sobel, gray=gray)

        if print_stats:
            print("double eval stats:")
            print(stats_dict2)
        else:
            config.double_eval_stats.append(stats_dict2)
            config.double_eval_acc.append(stats_dict2["eval_acc"])
            config.double_train_acc.append(stats_dict2["train_eval_acc"])

    net.eval()
    stats_dict = classify_head_eval_BCEloss(config, net, dataloader_eval=dataloader_eval,
                                            dataloader_train_eval=dataloader_train_eval,
                                            sobel=sobel, gray=gray)
    net.train()

    if print_stats:
        print("eval stats:")
        print(stats_dict)
    else:
        acc = stats_dict["eval_acc"]
        train_acc = stats_dict["train_eval_acc"]
        is_best = (len(config.epoch_acc) > 0) and (acc > max(config.epoch_acc))

        config.epoch_stats.append(stats_dict)
        config.epoch_acc.append(acc)
        config.epoch_train_acc.append(train_acc)

        return is_best
