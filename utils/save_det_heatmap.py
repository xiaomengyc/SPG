
import os
import cv2
import numpy as np


idx2catename = {'voc20': ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse',
                          'motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']}


def get_imgId(path_str):
    return path_str.strip().split('/')[-1].strip().split('.')[0]

def norm_atten_map(attention_map):
    min_val = np.min(attention_map)
    max_val = np.max(attention_map)
    atten_norm = (attention_map - min_val)/(max_val - min_val)
    return atten_norm*255

def add_colormap2img(img, atten_norm):
    heat_map = cv2.applycolormap(atten_norm.astype(np.uint8), cv2.colormap_jet)
    img = cv2.addweighted(img.astype(np.uint8), 0.5, heat_map.astype(np.uint8), 0.5, 0)
    return img

def save_atten(imgpath, atten, num_classes=20, base_dir='../save_bins/', idx_base=0):
    atten = np.squeeze(atten)
    for cls_idx in range(num_classes):
        cat_dir = os.path.join(base_dir, idx2catename['voc20'][cls_idx])
        if not os.path.exists(cat_dir): os.mkdir(cat_dir)
        cat_map = atten[cls_idx+idx_base]
        # read rgb image
        img = cv2.imread(imgpath)
        h, w, _ = np.shape(img)

        # reshape image
        cat_map = cv2.resize(cat_map, dsize=(w,h))
        cat_map = norm_atten_map(cat_map)

        # save heatmap
        save_path = os.path.join(cat_dir, get_imgId(imgpath)+'.png')
        cv2.imwrite(save_path, cat_map)
        # cv2.imwrite(save_path, add_colormap2img(img, cat_map))


def save_cls_scores(img_path, scores, base_dir='../save_bins/'):
    scores = np.squeeze(scores).tolist()
    score_str = map(lambda x:'%.4f'%(x), scores)

    with open(os.path.join(base_dir, 'scores.txt'), 'a') as fw:
        out_str = get_imgId(img_path) + ' ' + ' '.join(score_str) + '\n'
        fw.write(out_str)




