import argparse
from yacs.config import CfgNode as CN
import os.path as osp
from dataloader import get_splits
import cv2
import numpy as np
from time import time
from dataset.annotate import draw, get_dart_scores



def bboxes_to_xy(bboxes, max_darts=3):
    xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    for cls in range(5):
        if cls == 0:
            dart_xys = bboxes[bboxes[:, 4] == 0, :2][:max_darts]
            xy[4:4 + len(dart_xys), :2] = dart_xys
        else:
            cal = bboxes[bboxes[:, 4] == cls, :2]
            if len(cal):
                xy[cls - 1, :2] = cal[0]
    xy[(xy[:, 0] > 0) & (xy[:, 1] > 0), -1] = 1
    if np.sum(xy[:4, -1]) == 4:
        return xy
    else:
        xy = est_cal_pts(xy)
    return xy


def est_cal_pts(xy):
    missing_idx = np.where(xy[:4, -1] == 0)[0]
    if len(missing_idx) == 1:
        if missing_idx[0] <= 1:
            center = np.mean(xy[2:4, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 0:
                xy[0, 0] = -xy[1, 0]
                xy[0, 1] = -xy[1, 1]
            else:
                xy[1, 0] = -xy[0, 0]
                xy[1, 1] = -xy[0, 1]
            xy[:, :2] += center
        else:
            center = np.mean(xy[:2, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 2:
                xy[2, 0] = -xy[3, 0]
                xy[2, 1] = -xy[3, 1]
            else:
                xy[3, 0] = -xy[2, 0]
                xy[3, 1] = -xy[2, 1]
            xy[:, :2] += center
    else:
        # TODO: if len(missing_idx) > 1 (doesn't occur in dataset d1)
        print(missing_idx)
    return xy


def predict(
        yolo,
        cfg,
        labels_path='./dataset/labels.pkl',
        dataset='d1',
        split='val',
        max_darts=3):

    data = get_splits(labels_path, dataset, split)
    img_prefix = osp.join(cfg.data.path, 'cropped_images', str(cfg.model.input_size))
    img_paths = [osp.join(img_prefix, folder, name) for (folder, name) in zip(data.img_folder, data.img_name)]

    xys = np.zeros((len(data), 7, 3))  # third column for visibility
    data.xy = data.xy.apply(np.array)
    for i, _xy in enumerate(data.xy):
        xys[i, :_xy.shape[0], :2] = _xy
        xys[i, :_xy.shape[0], 2] = 1
    xys = xys.astype(np.float32)

    preds = np.zeros((len(img_paths), 4 + max_darts, 3))
    print('Making predictions...')

    for i, p in enumerate(img_paths):
        if i == 1:
            ti = time()
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes = yolo.predict(img)
        preds[i] = bboxes_to_xy(bboxes, max_darts)
        # img = draw(img, preds[i, :, :2], cfg, circles=True, score=True)
        # cv2.imshow('/'.join(p.split('/')[-2:]), img[:, :, [2, 1, 0]])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    print('FPS: {:.2f}'.format((len(img_paths) - 1) / (time() - ti)))

    ASE = []  # absolute score error
    for pred, gt in zip(preds, xys):
        ASE.append(abs(
            sum(get_dart_scores(pred[:, :2], cfg, numeric=True)) -
            sum(get_dart_scores(gt[:, :2], cfg, numeric=True))))
    ASE = np.array(ASE)
    print('Percent Correct Score (PSC): {:.1f}%'.format(len(ASE[ASE == 0]) / len(ASE) * 100))
    print('Mean Absolute Score Error (MASE): {:.2f}'.format(np.mean(ASE)))


if __name__ == '__main__':
    from train import build_model
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='tiny480_20e')
    parser.add_argument('-d', '--dataset', default='d1')
    parser.add_argument('-s', '--split', default='val')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))

    yolo = build_model(cfg)
    yolo.load_weights(osp.join('models', args.cfg, 'weights'), cfg.model.weights_type)

    predict(yolo, cfg, dataset=args.dataset, split=args.split)