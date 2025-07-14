import argparse
import os
import torch
from datetime import datetime
import sys
import cv2
import numpy as np

# pytorch_grad_cam 这个库被我修改过，以适用于我的模型，因为我的模型的输入是两张图像
from pytorch_grad_cam import (GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
                              AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
                              LayerCAM, FullGrad, GradCAMElementWise)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils.general import config_to_str, nice_2
from utils.dataloaders import create_dataloader_test
from utils.classify_test import classify_test_BCEloss
from networks.flownet_fc import flownet_fc_BCEloss
from utils.transforms import sobel_process, gray_process

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():

    # ------ options ------
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--model_id", type=int, default=2104)

    parser.add_argument("--dataset", type=str, default="03-external_val_dataset_2")
    parser.add_argument("--dataset_root", type=str,
                        default=r'./03-external_val_dataset_2/eval')  # "eval" or "train"

    parser.add_argument("--gt_k", type=int, default=1)
    parser.add_argument("--output_k", type=int, default=1)

    parser.add_argument("--batch_sz", type=int, default=8)  # num of pairs min-16 max-256

    parser.add_argument("--output_root", type=str, default=r'./test_visual_BCEloss')

    parser.add_argument("--double_eval", default=False, action="store_true")

    # visual
    parser.add_argument("--visual_cnn", dest="visual_cnn",
                        default=True, action="store_false")  # 是否进行cnn网络中间层学习情况的可视化输出
    parser.add_argument("--aug_smooth", default=True, action="store_false",
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', default=True, action='store_false',
                        help='Reduce noise by taking the first principle '
                             'component of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam', 'ablationcam',
                                 'eigencam', 'eigengradcam', 'layercam',
                                 'fullgrad', 'gradcamelementwise'], help='CAM method')

    # networks
    parser.add_argument("--well_trained_model_pth",
                        default=r'./well_trained_my_models/best_net.pytorch')
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

    assert (config.output_k == config.gt_k)

    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    print("Config: %s" % config_to_str(config))  # 将config配置参数转化为str，并print

    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise
    }

    # ------ main ------
    dataloader_test = create_dataloader_test(config)

    net = flownet_fc_BCEloss(config)
    net.load_state_dict(torch.load(config.well_trained_model_pth, map_location=lambda storage, loc: storage))
    net.cuda()
    # device_ids = [0, 1]
    # net = torch.nn.DataParallel(net, device_ids=device_ids)

    config.acc = []
    config.predictions = []
    config.predictions_soft = []
    config.gts = []
    config.img_pairs_name = []

    if config.double_eval:
        config.double_acc = []

    classify_test_BCEloss(config, net, dataloader_test=dataloader_test,
                          sobel=config.sobel_process, gray=config.gray_process)

    print("Test: time %s" % datetime.now())
    print("acc: %f" % config.acc[0])
    if config.double_eval:
        print("double eval acc: %f" % config.double_acc[0])
    for t in range(config.gts[0].shape[0]):
        if config.predictions_soft:
            print("\nimg_pair: %s" % config.img_pairs_name[t])
            print("gt: %d" % config.gts[0][t])
            print("prediction: %d" % config.predictions[0][t])
            print("prediction_soft: %f" % config.predictions_soft[0][t])
        else:
            print("\nimg_pair: %s" % config.img_pairs_name[t])
            print("gt: %d" % config.gts[0][t])
            print("prediction: %d" % config.predictions[0][t])

    sys.stdout.flush()

    # ------ visualization ------
    if config.visual_cnn:
        for b_i, batch in enumerate(dataloader_test):
            imgs_1 = batch[0].cuda()
            imgs_2 = batch[1].cuda()

            if config.sobel_process:
                imgs_1 = sobel_process(imgs_1, config.include_rgb)
                imgs_2 = sobel_process(imgs_2, config.include_rgb)

            if config.gray_process:
                imgs_1 = gray_process(imgs_1)
                imgs_2 = gray_process(imgs_2)

            img_pairs_name = batch[3]

            # 选择目标层
            # Choose the target layer you want to compute the visualization for.
            # Usually this will be the last  convolutional layer in the model.
            # You can print the model to help chose the layer.
            # You can pass a list with several target layers,
            # in that case the CAMs will be computed per layer and then aggregated.
            # You can also try selecting all layers of a certain type, with e.g:
            # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
            # find_layer_types_recursive(model, [torch.nn.ReLU])
            target_layer = [net.flownetc.conv6_1[-1]]

            # 选定目标类别，如果不设置，则默认为分数最高的那一类
            # We have to specify the target we want to generate the Class Activation Maps for.
            # If target_category is None,
            # the highest scoring category will be used for every image in the batch.
            # target category can be an integer,
            # or a list of different integers for every image in the batch.
            # You can target specific categories by
            # targets = [e.g. ClassifierOutputTarget(281)]
            target_category = None

            # 选择一个CAM算法，并初始化该对象
            # Using the 'with' statement ensures  the context is freed, and you can
            # recreate different CAM objects in a loop.
            # 译文：使用with语句可以确保释放上下文，并且您可以在循环中重新创建不同的CAM对象
            cam_algorithm = methods[config.method]
            with cam_algorithm(model=net, target_layers=target_layer, use_cuda=True) as cam:
                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = config.batch_sz

                # 计算cam
                # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                grayscale_cam = cam(input1_tensor=imgs_1, input2_tensor=imgs_2,
                                    targets=target_category,
                                    aug_smooth=config.aug_smooth,
                                    eigen_smooth=config.eigen_smooth)  # batch, h, w

                # 保存热力图，grayscale_cam是一个batch的结果
                for i in range(grayscale_cam.shape[0]):
                    grayscale_cam_one = grayscale_cam[i]

                    rgb_img_1 = batch[0][i].permute(1, 2, 0).numpy()
                    rgb_img_1 = np.float32(rgb_img_1) / 255
                    rgb_img_2 = batch[1][i].permute(1, 2, 0).numpy()
                    rgb_img_2 = np.float32(rgb_img_2) / 255

                    cam_image_1 = show_cam_on_image(rgb_img_1, grayscale_cam_one,
                                                    use_rgb=True)  # h, w, 3
                    cam_image_2 = show_cam_on_image(rgb_img_2, grayscale_cam_one,
                                                    use_rgb=True)

                    cam_image_1 = cv2.cvtColor(cam_image_1, cv2.COLOR_RGB2BGR)
                    cam_image_2 = cv2.cvtColor(cam_image_2, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(os.path.join(config.out_dir,
                                             "cam-" + img_pairs_name[i].split("&")[0]),
                                cam_image_1)
                    cv2.imwrite(os.path.join(config.out_dir,
                                             "cam-" + img_pairs_name[i].split("&")[1]),
                                cam_image_2)


if __name__ == "__main__":
    main()
