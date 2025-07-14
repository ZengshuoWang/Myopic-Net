import argparse
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import sys
from collections import OrderedDict

from utils.general import config_to_str, get_opt, nice, update_lr
from utils.dataloaders import create_dataloaders
from utils.classify_eval import classify_eval_BCEloss
from utils.transforms import sobel_process, gray_process
from networks.flownet_fc import flownet_fc_BCEloss

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


'''
Note: compared to 'train_eval.py', this code change the loss function 
from 'nn.CrossEntropyLoss()' to 'nn.BCEWithLogitsLoss()'. 
'''


def main():
    # ------ options ------
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--model_id", type=int, default=2104)
    parser.add_argument("--optimization", type=str, default="Adam")

    parser.add_argument("--dataset", type=str, default="MDCPS-crop-reduce-clean_orign_2")
    parser.add_argument("--dataset_root", type=str, default=r'./MDCPS-crop-reduce-clean_orign_2')

    parser.add_argument("--gt_k", type=int, default=1)
    parser.add_argument("--output_k", type=int, default=1)

    parser.add_argument("--lr", type=float, default=0.0001)  # 0.01
    parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
    parser.add_argument("--lr_mult", type=float, default=0.1)

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_sz", type=int, default=128)  # num of pairs min-16 max-256
    parser.add_argument("--num_dataloaders", type=int, default=1)  # 1 -> xx0x; 4 -> xx1x

    parser.add_argument("--output_root", type=str, default=r'./output_BCEloss')

    parser.add_argument("--restart", dest="restart", default=False, action="store_true")
    parser.add_argument("--restart_from_best", dest="restart_from_best",
                        default=False, action="store_true")

    parser.add_argument("--test_code", dest="test_code", default=False,
                        action="store_true")  # 测试代码的时候设为True

    parser.add_argument("--save_freq", type=int, default=20)

    parser.add_argument("--double_eval", default=False, action="store_true")

    parser.add_argument("--head_epochs", type=int, default=1)

    # networks
    parser.add_argument("--flownet_pth", default=r'./models/FlowNet2-C_checkpoint.pth.tar',
                        help="the well-trained flownet model")
    parser.add_argument("--fp16", action='store_true')

    # transforms
    parser.add_argument("--include_rgb", dest="include_rgb",
                        default=False, action="store_true")  # 在sobel处理输出时要不要在通道中包含rgb一起
    parser.add_argument("--sobel_process", dest="sobel_process",
                        default=False, action="store_true")  # 要不要对图像进行sobel处理

    parser.add_argument("--gray_process", dest="gray_process",
                        default=False, action="store_true")

    parser.add_argument("--input_size_h", type=int, default=384)
    parser.add_argument("--input_size_w", type=int, default=576)

    config = parser.parse_args()

    # ------ setup ------
    if config.sobel_process:
        if config.include_rgb:
            config.in_channels = 5
        else:
            config.in_channels = 2
    elif config.gray_process:
        config.in_channels = 1
    else:
        config.in_channels = 3

    config.out_dir = os.path.join(config.output_root, str(config.model_id))  # 输出路径
    assert (config.batch_sz % config.num_dataloaders == 0)
    config.dataloader_batch_sz = config.batch_sz / config.num_dataloaders

    assert (config.output_k == config.gt_k)

    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    if config.restart:  # 继续之前的训练
        config_name = "config.pickle"  # pickle是类似json的一种数据保存方式，也可以理解为一种文件保存类型
        net_name = "latest_net.pytorch"
        opt_name = "latest_optimiser.pytorch"

        if config.restart_from_best:
            config_name = "best_config.pickle"
            net_name = "best_net.pytorch"
            opt_name = "best_optimiser.pytorch"

        given_config = config
        reloaded_config_path = os.path.join(given_config.out_dir, config_name)  # 重新加载配置参数的路径
        print("Loading restarting config from: %s" % reloaded_config_path)
        with open(reloaded_config_path, "rb") as config_f:
            config = pickle.load(config_f)  # 加载路径下的配置参数
        assert (config.model_id == given_config.model_id)
        config.restart = True
        config.restart_from_best = given_config.restart_from_best

        # copy over new num_epochs and lr schedule
        config.num_epochs = given_config.num_epochs
        config.lr_schedule = given_config.lr_schedule

    else:
        print("Config: %s" % config_to_str(config))  # 将config配置参数转化为str，并print

    # ------ main ------
    dataloaders_train, dataloader_eval, dataloader_train_eval = create_dataloaders(config)

    # net = flownet_fc_BCEloss(config)  # batchNorm=False, 和flownet源代码中一致, 2024.03.17之前的结果都是在这种情况下跑出来的
    # 其实为了保持和flownet源码及预训练模型一致，就使用batchNorm=False就可以，
    # 这里就是想测试一下batchNorm=True的效果，需要注意的是在batchNorm=True的情况下，
    # flownet的结构变化相比原来还是挺大的，可以输出net_dict.items()里面的k以及pretrain_dict.items()里面的k对比一下
    net = flownet_fc_BCEloss(config, batchNorm=True)

    net_dict = net.state_dict()

    """
    for k, v in net_dict.items():
        print(k)
    """

    # 因此,flownet的预训练模型就是batchNorm=False的情况下的
    # 所以如果我们设置我们网络的batchNorm=True, 那么就需要注意flownet的预训练模型中缺少一些参数
    pretrained_dict = torch.load(config.flownet_pth)
    pretrained_dict = pretrained_dict['state_dict']

    """
    for k, v in pretrained_dict.items():
        print(k)
    """

    pretrained_dict_flownetc = OrderedDict({k: v for k, v in pretrained_dict.items()
                                            if ("flownetc." + k) in net_dict})
    # 因为net_dict中的模型关键词参数比flownetc中的模型关键词参数多了flownetc.的前缀，
    # 因此需要在搜索net_dict模型的时候需要在k的基础上添加上flownetc.的前缀
    # net_dict.update(pretrained_dict)
    # 因为同样的原因，不能直接使用字典数据的update()方法更新net_dict中的权重（即上面注释掉的这一句），
    # 而是需要自己写循环函数进行更新（见下面）
    keys_flownetc = []
    for k, v in net_dict.items():
        if (k.split('.', 1)[0] == 'flownetc') and (k.split('.', 1)[1] in pretrained_dict_flownetc):
            keys_flownetc.append(k)
    assert len(keys_flownetc) == len(pretrained_dict_flownetc)
    i = 0
    for k, v in pretrained_dict_flownetc.items():
        if v.size() == net_dict[keys_flownetc[i]].size():
            net_dict[keys_flownetc[i]] = pretrained_dict_flownetc[k]
            print(k + " in pretrained_dict_flownetc loading to " + keys_flownetc[i] + " in net successfully! ")
            i = i+1
        else:
            print("Load pre-trained weight wrong! ")
            assert v.size() == net_dict[keys_flownetc[i]].size()

    """
    # 下面是当输入数据的通道数不一定是3的时候的模型参数加载方式
    for k, v in pretrained_dict_flownetc.items():
        if v.size() == net_dict[keys_flownetc[i]].size():
            net_dict[keys_flownetc[i]] = pretrained_dict_flownetc[k]
            i = i + 1
        elif (v.size() != net_dict[keys_flownetc[i]].size()) and \
                (len(v.size()) == 4) and (len(net_dict[keys_flownetc[i]].size()) == 4):
            if (v.size()[1] > net_dict[keys_flownetc[i]].size()[1]) and \
                    (v[:, 0, :, :].size() == net_dict[keys_flownetc[i]][:, 0, :, :].size()):
                net_dict[keys_flownetc[i]][:, :, :, :] = \
                    pretrained_dict_flownetc[k][:, :net_dict[keys_flownetc[i]].size()[1], :, :]
                i = i + 1
            elif (v.size()[1] < net_dict[keys_flownetc[i]].size()[1]) and \
                    (v[:, 0, :, :].size() == net_dict[keys_flownetc[i]][:, 0, :, :].size()):
                net_dict[keys_flownetc[i]][:, :v.size()[1], :, :] = pretrained_dict_flownetc[k]
                net_dict[keys_flownetc[i]][:, v.size()[1]:, :, :] = \
                    pretrained_dict_flownetc[k][:, :(net_dict[keys_flownetc[i]].size()[1] - v.size()[1]), :, :]
                i = i + 1
            else:
                print(k + " in pretrained_dict_flownetc and " + keys_flownetc[i] + " in net are not match! ")
        else:
            assert ((v.size() == net_dict[keys_flownetc[i]].size()) or
                    ((len(v.size()) == 4) and (len(net_dict[keys_flownetc[i]].size()) == 4)))
    """

    net.load_state_dict(net_dict)

    if config.restart:
        model_path = os.path.join(config.out_dir, net_name)
        net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    net.cuda()
    device_ids = [0, 1]
    net = torch.nn.DataParallel(net, device_ids=device_ids)
    net.train()

    classify_loss = nn.BCEWithLogitsLoss().cuda()

    optimiser = get_opt(config.optimization)(net.module.parameters(), lr=config.lr)
    if config.restart:
        print("loading latest opt")
        optimiser.load_state_dict(torch.load(os.path.join(config.out_dir, opt_name)))

    head_epochs = config.head_epochs

    # ------ epoch 0 ------
    if config.restart:
        if not config.restart_from_best:
            next_epoch = config.last_epoch + 1  # corresponds to last saved model
        else:
            # sanity check
            next_epoch = np.argmax(np.array(config.epoch_acc)) + 1  # 找出精度最高的epoch
            assert (next_epoch == config.last_epoch + 1)  # 感觉这句话有问题?
        print("starting from epoch %d" % next_epoch)

        # in case we overshot without saving
        config.epoch_acc = config.epoch_acc[:next_epoch]  # in case we overshot
        config.epoch_train_acc = config.epoch_train_acc[:next_epoch]
        config.epoch_stats = config.epoch_stats[:next_epoch]

        if config.double_eval:
            config.double_eval_acc = config.double_eval_acc[:next_epoch]
            config.double_train_acc = config.double_train_acc[:next_epoch]
            config.double_eval_stats = config.double_eval_stats[:next_epoch]

        config.epoch_loss = config.epoch_loss[:(next_epoch - 1)]
    else:
        config.epoch_acc = []
        config.epoch_train_acc = []
        config.epoch_stats = []

        if config.double_eval:
            config.double_eval_acc = []
            config.double_train_acc = []
            config.double_eval_stats = []

        config.epoch_loss = []

        _ = classify_eval_BCEloss(config, net, dataloader_eval=dataloader_eval,
                                  dataloader_train_eval=dataloader_train_eval,
                                  sobel=config.sobel_process, gray=config.gray_process)

        print("Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
        if config.double_eval:
            print("double eval: \n %s" % (nice(config.double_eval_stats[-1])))
        sys.stdout.flush()
        next_epoch = 1  # 这个很重要，说明上面进行的是epoch0的训练

    fig, axarr = plt.subplots(3 + 2 * int(config.double_eval), sharex=False, figsize=(20, 20))

    # ------ train ------
    for e_i in range(next_epoch, config.num_epochs):
        print("Starting e_i: %d" % e_i)

        if e_i in config.lr_schedule:
            optimiser = update_lr(optimiser, lr_mult=config.lr_mult)  # 先更新学习率，因为上面已经训练过一个epoch0了

        avg_loss = 0.
        avg_loss_count = 0

        for head_epoch in range(head_epochs):
            sys.stdout.flush()

            iterators = (d for d in dataloaders_train)

            b_i = 0
            for tup in zip(*iterators):
                net.module.zero_grad()

                all_imgs_1 = torch.zeros((config.batch_sz, 3, config.input_size_h,
                                          config.input_size_w), requires_grad=True).cuda()
                all_imgs_2 = torch.zeros((config.batch_sz, 3, config.input_size_h,
                                          config.input_size_w), requires_grad=True).cuda()
                all_gts = torch.zeros(config.batch_sz, requires_grad=True).cuda()

                curr_batch_sz = 0
                for d_i in range(config.num_dataloaders):
                    imgs_1_curr = tup[d_i][0]
                    imgs_2_curr = tup[d_i][1]
                    gt_curr = tup[d_i][2]

                    curr_batch_sz = imgs_1_curr.size(0)

                    actual_batch_start = d_i * curr_batch_sz
                    actual_batch_end = actual_batch_start + curr_batch_sz

                    all_imgs_1[actual_batch_start:actual_batch_end, :, :, :] = imgs_1_curr.cuda()
                    all_imgs_2[actual_batch_start:actual_batch_end, :, :, :] = imgs_2_curr.cuda()
                    all_gts[actual_batch_start:actual_batch_end] = gt_curr.cuda()

                if not (curr_batch_sz == config.dataloader_batch_sz):
                    print("last batch sz %d" % curr_batch_sz)

                curr_total_batch_sz = curr_batch_sz * config.num_dataloaders
                # times 2
                all_imgs_1 = all_imgs_1[:curr_total_batch_sz, :, :, :]
                all_imgs_2 = all_imgs_2[:curr_total_batch_sz, :, :, :]
                all_gts = all_gts[:curr_total_batch_sz]

                if config.sobel_process:
                    all_imgs_1 = sobel_process(all_imgs_1, config.include_rgb)
                    all_imgs_2 = sobel_process(all_imgs_2, config.include_rgb)

                if config.gray_process:
                    all_imgs_1 = gray_process(all_imgs_1)
                    all_imgs_2 = gray_process(all_imgs_2)

                x_outs = net(all_imgs_1, all_imgs_2)
                x_outs = x_outs.squeeze(dim=-1)

                loss = classify_loss(x_outs, all_gts)

                if ((b_i % 100) == 0) or (e_i == next_epoch and b_i < 10):
                    print("Model id %d epoch %d head_epoch %d batch %d: avg loss %f time %s" %
                          (config.model_id, e_i, head_epoch, b_i, loss.item(), datetime.now()))
                    sys.stdout.flush()

                if not np.isfinite(loss.item()):
                    print("Loss is not finite... %s:" % loss.item())
                    exit(1)

                avg_loss += loss.item()
                avg_loss_count += 1

                loss.backward()
                optimiser.step()

                b_i += 1
                if b_i == 2 and config.test_code:
                    break

        avg_loss = float(avg_loss / avg_loss_count)

        config.epoch_loss.append(avg_loss)

        # ------ eval ------
        is_best = classify_eval_BCEloss(config, net, dataloader_eval=dataloader_eval,
                                        dataloader_train_eval=dataloader_train_eval,
                                        sobel=config.sobel_process, gray=config.gray_process)

        print("Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
        if config.double_eval:
            print("double eval: \n %s" % (nice(config.double_eval_stats[-1])))
        sys.stdout.flush()

        axarr[0].clear()
        axarr[0].plot(config.epoch_acc)
        axarr[0].set_title("acc (eval), top: %f" % max(config.epoch_acc))

        axarr[1].clear()
        axarr[1].plot(config.epoch_train_acc)
        axarr[1].set_title("acc (train), top: %f" % max(config.epoch_train_acc))

        axarr[2].clear()
        axarr[2].plot(config.epoch_loss)
        axarr[2].set_title("Loss")

        if config.double_eval:
            axarr[3].clear()
            axarr[3].plot(config.double_eval_acc)
            axarr[3].set_title("double eval acc (best), top: %f" % max(config.double_eval_acc))

            axarr[4].clear()
            axarr[4].plot(config.double_train_acc)
            axarr[4].set_title("double train acc (best), top: %f" % max(config.double_train_acc))

        fig.tight_layout()
        fig.canvas.draw_idle()
        fig.savefig(os.path.join(config.out_dir, "plots.png"))

        if is_best or (e_i % config.save_freq == 0):
            net.module.cpu()

            if e_i % config.save_freq == 0:
                torch.save(net.module.state_dict(), os.path.join(config.out_dir, "latest_net.pytorch"))
                torch.save(optimiser.state_dict(), os.path.join(config.out_dir, "latest_optimiser.pytorch"))

                config.last_epoch = e_i  # for last saved version

            if is_best:
                # also serves as backup if hardware fails - less likely to hit this
                torch.save(net.module.state_dict(), os.path.join(config.out_dir, "best_net.pytorch"))
                torch.save(optimiser.state_dict(), os.path.join(config.out_dir, "best_optimiser.pytorch"))

                with open(os.path.join(config.out_dir, "best_config.pickle"), 'wb') as outfile:
                    pickle.dump(config, outfile)

                with open(os.path.join(config.out_dir, "best_config.txt"), "w") as text_file:
                    text_file.write("%s" % config)

            net.module.cuda()

        with open(os.path.join(config.out_dir, "config.pickle"), 'wb') as outfile:
            pickle.dump(config, outfile)

        with open(os.path.join(config.out_dir, "config.txt"), "w") as text_file:
            text_file.write("%s" % config)

        if config.test_code:
            exit(0)


if __name__ == "__main__":
    main()
