import argparse
from yacs.config import CfgNode as CN
import os.path as osp
from dataloader import get_splits
import cv2
import numpy as np
from dataset.annotate import draw


def bboxes_to_xy(bboxes, max_darts=3):
    xy = np.zeros((4 + max_darts, 2), dtype=np.float32)
    for cls in range(5):
        if cls == 0:
            dart_xys = bboxes[bboxes[:, 4] == 0, :2][:max_darts]
            xy[4:4 + len(dart_xys)] = dart_xys
        else:
            cal = bboxes[bboxes[:, 4] == cls, :2]
            if len(cal):
                xy[cls - 1] = cal[0]
    return xy


def preview(
        yolo,
        cfg,
        labels_path='./dataset/labels.pkl',
        dataset='d1',
        split='val',
):

    data = get_splits(labels_path, dataset, split)
    img_prefix = osp.join(cfg.data.path, 'cropped_images', str(cfg.model.input_size))
    img_prefix_large = osp.join(cfg.data.path, 'cropped_images', '800')
    img_paths = [osp.join(img_prefix, folder, name) for (folder, name) in zip(data.img_folder, data.img_name)]
    img_paths_large = [osp.join(img_prefix_large, folder, name) for (folder, name) in zip(data.img_folder, data.img_name)]

    for p, pf in zip(img_paths, img_paths_large):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes = yolo.predict(img)
        xy = bboxes_to_xy(bboxes)
        xy = xy[(xy[:, 0] > 0) & (xy[:, 1] > 0)]

        img_full = cv2.imread(pf)
        img = draw(img_full, xy, cfg, circles=True, score=True)

        cv2.imshow('/'.join(p.split('/')[-2:]), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    from train import build_model
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='tiny480_20e_rot18')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))

    yolo = build_model(cfg)
    yolo.load_weights(osp.join('models', args.cfg, 'weights'), cfg.model.weights_type)

    preview(yolo, cfg)

