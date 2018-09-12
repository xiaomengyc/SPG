import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import os
import cv2
import numpy as np


__all__ = ['Inception3', 'inception_v3']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


def model(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception3(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
        return model

    return Inception3(**kwargs)

class SPP_A(nn.Module):
    def __init__(self, in_channels, rates = [1,3,6]):
        super(SPP_A, self).__init__()
        self.aspp = []
        for r in rates:
            self.aspp.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=128, kernel_size=3, dilation=r, padding=r),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, out_channels=128, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
            )
        self.out_conv1x1 = nn.Conv2d(128*len(rates), 1, kernel_size=1)

    def forward(self, x):
        aspp_out = torch.cat([classifier(x) for classifier in self.aspp], dim=1)
        return self.out_conv1x1(aspp_out)

class SPP_B(nn.Module):
    def __init__(self, in_channels, num_classes=1000, rates = [1,3,6]):
        super(SPP_B, self).__init__()
        self.aspp = []
        for r in rates:
            self.aspp.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=1024, kernel_size=3, dilation=r, padding=r),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(1024, out_channels=1024, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
            )
        self.out_conv1x1 = nn.Conv2d(1024, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        aspp_out = torch.mean([classifier(x) for classifier in self.aspp], dim=1)
        return self.out_conv1x1(aspp_out)

class Inception3(nn.Module):

    def __init__(self, num_classes=1000, args=None, threshold=None, transform_input=False):
        super(Inception3, self).__init__()
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.num_classes = args.num_classes

        #Added
        self.fc6 = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),   #fc6
            nn.ReLU(True),
        )
        self.th = threshold
        self.fc7_1 = self.apc(1024, num_classes, kernel=3, rate=1)
        self.classier_1 = nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)  #fc8

        # Branch B
        self.branchB = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 1, kernel_size=1)
        )

        #------------------------------------------
        #Segmentation
        self.side3 = self.side_cls(288, kernel_size=3, padding=1)
        self.side4 = self.side_cls(768, kernel_size=3, padding=1)

        self.side_all = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1, padding=0, dilation=1),
        )

        self.interp = nn.Upsample(size=(224,224), mode='bilinear')
        self._initialize_weights()
        self.loss_cross_entropy = nn.CrossEntropyLoss()
        # self.loss_func = nn.CrossEntropyLoss(ignore_index=255)
        self.loss_func = nn.BCEWithLogitsLoss()

    def side_cls(self, in_planes, kernel_size=3, padding=1 ):
        return nn.Sequential(
            nn.Conv2d(in_planes, 512, kernel_size=kernel_size, padding=padding, dilation=1),
            nn.ReLU(inplace=True),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x, label=None):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 224 x 224 x 3
        x = self.Conv2d_1a_3x3(x)
        # 112 x 112 x 32
        x = self.Conv2d_2a_3x3(x)
        # 112 x 112 x 32
        x = self.Conv2d_2b_3x3(x)
        # 112 x 112 x 32
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        # 56 x 56 x 64
        x = self.Conv2d_3b_1x1(x)
        # 56 x 56 x 64
        x = self.Conv2d_4a_3x3(x)
        # 56 x 56 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        # 28 x 28 x 192
        x = self.Mixed_5b(x)
        # 28 x 28 x 192
        x = self.Mixed_5c(x)
        # 28 x 28 x 192
        x = self.Mixed_5d(x)

        side3 = self.side3(x)
        side3 = self.side_all(side3)

        # 28 x 28 x 192
        x = self.Mixed_6a(x)
        # 28 x 28 x 768
        x = self.Mixed_6b(x)
        # 28 x 28 x 768
        x = self.Mixed_6c(x)
        # 28 x 28 x 768
        x = self.Mixed_6d(x)
        # 28 x 28 x 768
        feat = self.Mixed_6e(x)

        side4 = self.side4(x)
        side4 = self.side_all(side4)

        #Branch 1
        out1, last_feat = self.inference(feat, label=label)
        self.map1 = out1

        atten_map = self.get_atten_map(self.interp(out1), label, True)

        #Branch B
        out_seg = self.branchB(last_feat)

        logits_1 = torch.mean(torch.mean(out1, dim=2), dim=2)

        return [logits_1, side3, side4, out_seg, atten_map]

    def inference(self, x, label=None):
        x = F.dropout(x, 0.5)
        x = self.fc6(x)
        x = F.dropout(x, 0.5)
        x = self.fc7_1(x)
        x = F.dropout(x, 0.5)
        out1 = self.classier_1(x)
        return out1, x

    def apc(self, in_planes=1024, out_planes=1024, kernel=3, rate=1):
        return nn.Sequential(
            nn.Conv2d(in_planes, 1024, kernel_size=kernel, padding=rate, dilation=rate),   #fc6
            nn.ReLU(True),
            # nn.Conv2d(1024, out_planes, kernel_size=3, padding=1)  #fc8
        )

    def mark_obj(self, label_img, heatmap, label, threshold=0.5):

        if isinstance(label, (float, int)):
            np_label = label
        else:
            np_label = label.cpu().data.numpy().tolist()

        for i in range(heatmap.size()[0]):
            mask_pos = heatmap[i] > threshold
            if torch.sum(mask_pos.float()).data.cpu().numpy() < 30:
                threshold = torch.max(heatmap[i]) * 0.7
                mask_pos = heatmap[i] > threshold
            label_i = label_img[i]
            if isinstance(label, (float, int)):
                use_label = np_label
            else:
                use_label = np_label[i]
            # label_i.masked_fill_(mask_pos.data, use_label)
            label_i[mask_pos.data] = use_label
            label_img[i] = label_i

        return label_img

    def mark_bg(self, label_img, heatmap, threshold=0.1):
        mask_pos = heatmap < threshold
        # label_img.masked_fill_(mask_pos.data, 0.0)
        label_img[mask_pos.data] = 0.0

        return label_img

    def get_mask(self, mask, atten_map, th_high=0.7, th_low = 0.05):
        #mask label for segmentation
        mask = self.mark_obj(mask, atten_map, 1.0, th_high)
        mask = self.mark_bg(mask, atten_map, th_low)

        return  mask

    def get_loss(self, logits, gt_labels):
        logits_1, side3, side4, out_seg, atten_map = logits

        loss_cls = self.loss_cross_entropy(logits_1, gt_labels.long())

        # atten_map = logits[-1]
        mask = torch.zeros((logits_1.size()[0], 224, 224)).fill_(255).cuda()
        mask = self.get_mask(mask, atten_map)

        mask_side4 = torch.zeros((logits_1.size()[0], 224, 224)).fill_(255).cuda()
        mask_side4 = self.get_mask(mask_side4, torch.squeeze(F.sigmoid(self.interp(side4))), 0.5, 0.05)

        loss_side4 = self.loss_saliency(self.loss_func, self.interp(side4).squeeze(dim=1), mask)
        loss_side3 = self.loss_saliency(self.loss_func, self.interp(side3).squeeze(dim=1), mask_side4)

        fused_atten = (F.sigmoid(self.interp(side3)) + F.sigmoid(self.interp(side4)))/2.0

        back_mask = torch.zeros((logits_1.size()[0], 224, 224)).fill_(255).cuda()
        back_mask = self.get_mask(back_mask, torch.squeeze(fused_atten.detach()), 0.7, 0.1)
        loss_back = self.loss_saliency(self.loss_func, self.interp(out_seg).squeeze(dim=1), back_mask)

        loss_val = loss_cls + loss_side3 + loss_side4 + loss_back
        return [loss_val, ]

    def loss_saliency(self, loss_func, logtis, labels):
        positions = labels.view(-1, 1) < 255.0
        return loss_func(logtis.view(-1, 1)[positions], Variable(labels.view(-1, 1)[positions]))


    def loss_segmentation(self, loss_func, logits, labels):
        logits = logits.permute(0, 2, 3, 1).contiguous().view((-1, self.num_classes+1))
        labels = labels.view(-1).long()

        return loss_func(logits, Variable(labels))

    def loss_erase_step(self, logits, gt_labels):
        loss_val = 0
        # for single label images
        logits = F.softmax(logits, dim=1)

        if len(logits.size()) != len(gt_labels.size()):

            for batch_idx in range(logits.size()[0]):
                loss_val += logits[batch_idx, gt_labels.cpu().data.numpy()[batch_idx]]

        #try torch.sum(logits[:,gt_labels])

        return loss_val/(logits.size()[0])

    def save_erased_img(self, img_path, img_batch=None):
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
        if img_batch is None:
            img_batch = self.img_erased
        if len(img_batch.size()) == 4:
            batch_size = img_batch.size()[0]
            for batch_idx in range(batch_size):
                imgname = img_path[batch_idx]
                nameid = imgname.strip().split('/')[-1].strip().split('.')[0]

                # atten_map = F.upsample(self.attention.unsqueeze(dim=1), (321,321), mode='bilinear')
                atten_map = F.upsample(self.attention.unsqueeze(dim=1), (224,224), mode='bilinear')
                # atten_map = F.upsample(self.attention, (224,224), mode='bilinear')
                # mask = F.sigmoid(20*(atten_map-0.5))
                mask = atten_map
                mask = mask.squeeze().cpu().data.numpy()

                img_dat = img_batch[batch_idx]
                img_dat = img_dat.cpu().data.numpy().transpose((1,2,0))
                img_dat = (img_dat*std_vals + mean_vals)*255

                img_dat = self.add_heatmap2img(img_dat, mask)
                save_path = os.path.join('../save_bins/', nameid+'.png')
                cv2.imwrite(save_path, img_dat)
                # save_path = os.path.join('../save_bins/', nameid+'..png')
                # cv2.imwrite(save_path, mask*255)


    def add_heatmap2img(self, img, heatmap):
        # assert np.shape(img)[:3] == np.shape(heatmap)[:3]

        heatmap = heatmap* 255
        color_map = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        img_res = cv2.addWeighted(img.astype(np.uint8), 0.5, color_map.astype(np.uint8), 0.5, 0)

        return img_res

    def get_localization_maps(self):
        map1 = self.normalize_atten_maps(self.map1)
        # map_erase = self.normalize_atten_maps(self.map_erase)
        # return torch.max(map1, map_erase)
        return map1

    def get_feature_maps(self):
        return self.normalize_atten_maps(self.map1)

    def get_heatmaps(self, gt_label):
        map1 = self.get_atten_map(self.map1, gt_label)
        return [map1,]

    def get_fused_heatmap(self, gt_label):
        maps = self.get_heatmaps(gt_label=gt_label)
        fuse_atten = maps[0]
        return fuse_atten

    def get_maps(self, gt_label):
        map1 = self.get_atten_map(self.map1, gt_label)
        return [map1, ]

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

    def get_atten_map(self, feature_maps, gt_labels, normalize=True):
        label = gt_labels.long()

        feature_map_size = feature_maps.size()
        batch_size = feature_map_size[0]

        atten_map = torch.zeros([feature_map_size[0], feature_map_size[2], feature_map_size[3]])
        atten_map = Variable(atten_map.cuda())
        for batch_idx in range(batch_size):
            atten_map[batch_idx,:,:] = torch.squeeze(feature_maps[batch_idx, label.data[batch_idx], :,:])

        if normalize:
            atten_map = self.normalize_atten_maps(atten_map)

        return atten_map


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, kernel_size=3, stride=2, padding=0):
        self.stride = stride
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384,
                                     kernel_size=kernel_size, stride=stride, padding=padding)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=stride, padding=padding)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
