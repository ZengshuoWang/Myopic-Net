from torch.nn import init
from .correlation_package.correlation import Correlation
from .sub_modules import *
import torch


class FlowNetC(nn.Module):
    def __init__(self, config, batchNorm=True):
        super(FlowNetC, self).__init__()

        self.batchNorm = batchNorm

        self.conv1 = conv(self.batchNorm, config.in_channels, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1)

        if config.fp16:
            self.corr = nn.Sequential(
                tofp32(),
                Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1),
                tofp16()
            )
        else:
            self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20,
                                    stride1=1, stride2=2, corr_multiply=1)

        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv3_1 = conv(self.batchNorm, 473, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, image1, image2):
        # 提取第一幅图像的特征
        out_conv1a = self.conv1(image1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # 提取第二幅图像的特征
        out_conv1b = self.conv1(image2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # 合并流（计算两幅特征图的相关性）
        out_corr = self.corr(out_conv3a, out_conv3b)
        out_corr = self.corr_activation(out_corr)

        # 再对聚焦图像做一次卷积，然后连接到合并流的后面
        out_conv_redir = self.conv_redir(out_conv3a)
        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # 对合并后的流进行一系列的卷积运算
        out_conv3_1 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        # 到此为止会让图像的尺寸缩小至输入图像的64分之一

        return out_conv6


class FC_BCEloss(nn.Module):
    def __init__(self, config, input_features_channels, output_k):
        super(FC_BCEloss, self).__init__()

        num_features = input_features_channels
        features_sp_size_h = int(config.input_size_h / 64)
        features_sp_size_w = int(config.input_size_w / 64)

        '''
        # output -> 1xxx
        net_head = nn.Sequential(
            nn.Linear(num_features * features_sp_size_h * features_sp_size_w, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, output_k)
        )
        '''
        # output -> 2xxx
        net_head = nn.Sequential(
            nn.Linear(num_features * features_sp_size_h * features_sp_size_w, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_k)
        )
        '''
        # output -> 3xxx
        net_head = nn.Sequential(
            nn.Linear(num_features * features_sp_size_h * features_sp_size_w, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_k)
        )
        '''

        net_head.cuda()

        self.head = net_head

        for m in self.head.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.head(x)  # with no softmax

        return x


class FC_BCEloss_AvgPool(nn.Module):
    def __init__(self, config, input_features_channels, output_k):
        super(FC_BCEloss_AvgPool, self).__init__()

        num_features = input_features_channels

        # output -> 2xxx
        net_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_k)
        )

        net_head.cuda()

        self.head = net_head

        for m in self.head.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        avg_pool = nn.AdaptiveAvgPool2d(1)
        x = avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.head(x)  # with no softmax

        return x


class flownet_fc_BCEloss(nn.Module):
    def __init__(self, config, batchNorm=False):  # flownet源代码中用的就是batchNorm=False
        super(flownet_fc_BCEloss, self).__init__()

        self.batchNorm = batchNorm

        self.flownetc = FlowNetC(config, batchNorm=self.batchNorm)
        self.fc = FC_BCEloss(config, input_features_channels=1024, output_k=config.output_k)

    def forward(self, image1, image2):
        # 输入图像归一化
        image1 = 2 * (image1 / 255.0) - 1.0  # 图像归一化
        image2 = 2 * (image2 / 255.0) - 1.0  # 图像归一化

        image1 = image1.contiguous()  # 感觉类似进行了一次深拷贝
        image2 = image2.contiguous()

        x = self.flownetc(image1, image2)
        x = self.fc(x)

        return x


class flownet_fc_BCEloss_Normal1(nn.Module):
    def __init__(self, config, batchNorm=False):  # flownet源代码中用的就是batchNorm=False
        super(flownet_fc_BCEloss_Normal1, self).__init__()

        self.batchNorm = batchNorm

        self.flownetc = FlowNetC(config, batchNorm=self.batchNorm)
        self.fc = FC_BCEloss(config, input_features_channels=1024, output_k=config.output_k)

    def forward(self, image1, image2):
        # 输入图像归一化
        rgb_mean_1 = image1.contiguous().view(image1.size()[:2]+(-1,)).mean(dim=-1).view(image1.size()[:2]+(1, 1,))
        rgb_mean_2 = image2.contiguous().view(image2.size()[:2]+(-1,)).mean(dim=-1).view(image2.size()[:2]+(1, 1,))
        image1 = (image1 - rgb_mean_1) / 255.
        image2 = (image2 - rgb_mean_2) / 255.

        image1 = image1.contiguous()  # 感觉类似进行了一次深拷贝
        image2 = image2.contiguous()

        x = self.flownetc(image1, image2)
        x = self.fc(x)

        return x


class flownet_fc_BCEloss_AvgPool(nn.Module):
    def __init__(self, config, batchNorm=False):  # flownet源代码中用的就是batchNorm=False
        super(flownet_fc_BCEloss_AvgPool, self).__init__()

        self.batchNorm = batchNorm

        self.flownetc = FlowNetC(config, batchNorm=self.batchNorm)
        self.fc = FC_BCEloss_AvgPool(config, input_features_channels=1024, output_k=config.output_k)

    def forward(self, image1, image2):
        # 输入图像归一化
        image1 = 2 * (image1 / 255.0) - 1.0  # 图像归一化
        image2 = 2 * (image2 / 255.0) - 1.0  # 图像归一化

        image1 = image1.contiguous()  # 感觉类似进行了一次深拷贝
        image2 = image2.contiguous()

        x = self.flownetc(image1, image2)
        x = self.fc(x)

        return x
