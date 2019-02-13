# general includes
import os, sys
import argparse
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from collections import OrderedDict
from copy import deepcopy
import re
import skvideo.io

# pytorch includes
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# custom includes
import models.google.inception3_spg as inceptionv3_spg

# some general settings
mean_vals = [0.485, 0.456, 0.406]
std_vals = [0.229, 0.224, 0.225]

# read ImageNet labels
imagenet_label = np.loadtxt('map_clsloc.txt', str, delimiter='\t')

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    def atoi(text):
        return int(text) if text.isdigit() else text

    return [ atoi(c) for c in re.split('(\d+)', text) ]

def get_arguments():
    parser = argparse.ArgumentParser(description='SPG')
    parser.add_argument('--video_dir', type=str, default='examples/')
    parser.add_argument('--save_dir', type=str, default='examples/results/')
    parser.add_argument('--video_name', type=str, default=None)
    parser.add_argument('--input_size', type=int, default=321)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--snapshots', type=str, default='snapshots/imagenet_epoch_2_glo_step_128118.pth.tar')
    parser.add_argument('--heatmap_type', type=str, default='loc')
    parser.add_argument('--include_ori', type=bool, default=True)
    parser.add_argument('--high_res', type=bool, default=False)
    return parser.parse_args()

def load_model(args):
    model = inceptionv3_spg.Inception3(num_classes=args.num_classes, threshold=0.5)
    checkpoint = torch.load(args.snapshots)

    # to match the names in pretrained model
    pretrained_dict = OrderedDict()
    for ki in checkpoint['state_dict'].keys():
        pretrained_dict[ki[7:]] = deepcopy(checkpoint['state_dict'][ki])
    model.load_state_dict(pretrained_dict)
    print('Loaded checkpoint: {}'.format(args.snapshots))

    return model


if __name__ == '__main__':
    args = get_arguments()
    print(args)

    # setup model
    model = load_model(args)
    model = model.cuda()
    model = model.eval()

    # read videos
    video_dir = os.path.join(args.video_dir, args.video_name)
    img_list = os.listdir(video_dir)
    img_list.sort(key=natural_keys)
    save_name = os.path.join(args.save_dir, args.video_name + '-heatmap-{}.avi'.format(args.heatmap_type))
    if args.high_res:
        writer = skvideo.io.FFmpegWriter(save_name,  outputdict={'-b': '300000000'})
    else:
        writer = skvideo.io.FFmpegWriter(save_name)


    for name in img_list:
        image = np.float32(Image.open(os.path.join(video_dir, name)))
        img_h, img_w = image.shape[:2]

        # pre-process image
        inputs = cv2.resize(image, (args.input_size, args.input_size))
        inputs = torch.from_numpy(inputs.copy().transpose((2, 0, 1))).float().div(255)
        for t, m, s in zip(inputs, mean_vals, std_vals):
            t.sub_(m).div_(s) # de-mean
        inputs = inputs.unsqueeze(0) # add batch dimension

        # forward pass
        inputs = Variable(inputs).cuda()
        label = Variable(torch.zeros(1).long()).cuda() # dummy variable
        # the outputs are (0) logits (1000), (1) side3, (2) side4, (4) out_seg (SPG-C) and (5) atten_map
        # remarks: do not use output (5)
        outputs = model(inputs, label)

        # obtain top k classification results
        logits = outputs[0]
        logits = F.softmax(logits, dim=1)
        np_scores, pred_labels = torch.topk(logits, k=1, dim=1)
        np_pred_labels = pred_labels.cpu().data.numpy()[0][0]

        if args.heatmap_type == 'loc':
            # obtain heatmaps for all classes
            last_featmaps = model.get_localization_maps()
            np_last_featmaps = last_featmaps.cpu().data.numpy()
            attention_map = np_last_featmaps[0, np_pred_labels, :]

        elif args.heatmap_type == 'spg_c':
            # use heatmap from the SPG-C branch
            out_seg = outputs[3]
            attention_map = F.sigmoid(out_seg).cpu().data.numpy()[0, 0, :]

        else:
            raise ValueError('Unknown heatmap type: {}'.format(args.heatmap_type))

        # normalize attention map
        atten_norm = cv2.resize(attention_map, dsize=(img_w, img_h)) * 255
        heatmap = cv2.applyColorMap(np.uint8(atten_norm), cv2.COLORMAP_JET)
        out = cv2.addWeighted(np.uint8(image), 0.5, np.uint8(heatmap), 0.5, 0)

        # save text
        pred_class = imagenet_label[np_pred_labels].split(' ')[-1]
        position = (50, img_h - 50)
        cv2.putText(out, pred_class, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out = out[:, :, ::-1] # BGR (opencv) -> RGB

        # show original image side-by-side?
        if args.include_ori:
            save_image = np.ones((img_h, 2*img_w + 10, 3)) * 255
            save_image[:, :img_w, :] = image
            save_image[:, img_w+10:, :] = out
        else:
            save_image = out

        writer.writeFrame(save_image)
    writer.close()