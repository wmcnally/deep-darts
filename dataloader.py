import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os.path as osp
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from dataset.annotate import draw, transform
from yacs.config import CfgNode as CN
from yolov4.tf.dataset import cut_out


d1_val = ['d1_02_06_2020', 'd1_02_16_2020', 'd1_02_22_2020']
d1_test = ['d1_03_03_2020', 'd1_03_19_2020', 'd1_03_23_2020', 'd1_03_27_2020', 'd1_03_28_2020', 'd1_03_30_2020', 'd1_03_31_2020']

d2_val = ['d2_02_03_2021', 'd2_02_05_2021']
d2_test = ['d2_03_03_2020', 'd2_02_10_2021', 'd2_02_03_2021_2']


def get_splits(path='./dataset/labels.pkl', dataset='d1', split='train'):
    assert dataset in ['d1', 'd2'], "dataset must be either 'd1' or 'd2'"
    assert split in [None, 'train', 'val', 'test'], "split must be in [None, 'train', 'val', 'test']"
    if dataset == 'd1':
        val_folders, test_folders = d1_val, d1_test
    else:
        val_folders, test_folders = d2_val, d2_test
    df = pd.read_pickle(path)
    df = df[df.img_folder.str.contains(dataset)]
    splits = {}
    splits['val'] = df[np.isin(df.img_folder, val_folders)]
    splits['test'] = df[np.isin(df.img_folder, test_folders)]
    splits['train'] = df[np.logical_not(np.isin(df.img_folder, val_folders + test_folders))]
    if split is None:
        return splits
    else:
        return splits[split]


def preprocess(path, xy, cfg, bbox_to_gt_func, split='train', return_xy=False):
    path = path.numpy().decode('utf-8')
    xy = xy.numpy()

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # yolov4 tf convention
    img = img / 255.  # yolov4 tf convention

    if split == 'train' and np.random.uniform() < cfg.aug.overall_prob:
        transformed = False

        if cfg.aug.flip_lr_prob and np.random.uniform() < cfg.aug.flip_lr_prob:
            if not transformed:
                xy, img, M = transform(xy, img)
                transformed = True
            img, xy = flip(img, xy, direction='lr')

        if cfg.aug.flip_ud_prob and np.random.uniform() < cfg.aug.flip_ud_prob:
            if not transformed:
                xy, img, M = transform(xy, img)
                transformed = True
            img, xy = flip(img, xy, direction='ud')

        if cfg.aug.rot_prob and np.random.uniform() < cfg.aug.rot_prob:
            if not transformed:
                xy, img, M = transform(xy, img)
                transformed = True
            angles = np.arange(-180, 180, step=cfg.aug.rot_step)
            angle = angles[np.random.randint(len(angles))]
            img, xy = rotate(img, xy, angle, darts_only=True)

        if cfg.aug.rot_small_prob and np.random.uniform() < cfg.aug.rot_small_prob:
            angle = np.random.uniform(-cfg.aug.rot_small_max, cfg.aug.rot_small_max)
            img, xy = rotate(img, xy, angle, darts_only=False)  # rotate cal points too

        if cfg.aug.jitter_prob and np.random.uniform() < cfg.aug.jitter_prob:
            h, w = img.shape[:2]
            jitter = cfg.aug.jitter_max * w
            tx = np.random.uniform(-1, 1) * jitter
            ty = np.random.uniform(-1, 1) * jitter
            img, xy = translate(img, xy, tx, ty)

        if cfg.aug.warp_prob and np.random.uniform() < cfg.aug.warp_prob:
            if not transformed:
                xy, img, M = transform(xy, img)
            M_inv = np.linalg.inv(M)
            M_inv[0, 1:3] *= np.random.uniform(0, cfg.aug.warp_rho, 2)
            M_inv[1, [0, 2]] *= np.random.uniform(0, cfg.aug.warp_rho, 2)
            M_inv[2, 0:2] *= np.random.uniform(0, cfg.aug.warp_rho, 2)
            xy, img, _ = transform(xy, img, M=M_inv)

        else:
            if transformed:
                M_inv = np.linalg.inv(M)
                xy, img, _ = transform(xy, img, M=M_inv)

    if return_xy:
        return img, xy

    bboxes = get_bounding_boxes(xy, cfg.train.bbox_size)

    if split == 'train':
        # cutout augmentation
        if cfg.aug.cutout_prob and np.random.uniform() < cfg.aug.cutout_prob:
            img, bboxes = cut_out([np.expand_dims(img, axis=0), bboxes])
            img = img[0]

    gt = bbox_to_gt_func(bboxes)
    gt = [item.squeeze() for item in gt]
    return (img, *gt)


def align_board(img, xy):
    center = np.mean(xy[:4, :2], axis=0)
    angle = 9 - np.arctan((center[0] - xy[0, 0]) / (center[1] - xy[0, 1])) / np.pi * 180
    img, xy = rotate(img, xy, angle, darts_only=False)
    return img, xy


def rotate(img, xy, angle, darts_only=True):
    h, w = img.shape[:2]
    center = np.mean(xy[:4, :2], axis=0)
    M = cv2.getRotationMatrix2D((center[0]*w, center[1]*h), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))

    vis = xy[:, 2:]
    xy = xy[:, :2]
    if darts_only:
        if xy.shape[0] > 4:
            xy_darts = xy[4:]
            xy_darts -= center
            xy_darts = np.matmul(M[:, :2], xy_darts.T).T
            xy_darts += center
            xy[4:] = xy_darts
    else:
        xy -= center
        xy = np.matmul(M[:, :2], xy.T).T
        xy += center
    xy = np.concatenate([xy, vis], axis=-1)
    return img, xy


def flip(img, xy, direction, darts_only=True):
    if direction == 'lr':
        img = img[:, ::-1, :]  # flip left-right
        axis = 0
    else:
        img = img[::-1, :, :]  # flip up-down
        axis = 1
    center = np.mean(xy[:4, :2], axis=0)
    vis = xy[:, 2:]
    xy = xy[:, :2]
    if darts_only:
        if xy.shape[0] > 4:
            xy_darts = xy[4:]
            xy_darts -= center
            xy_darts[:, axis] = -xy_darts[:, axis]
            xy_darts += center
            xy[4:] = xy_darts
    else:
        xy -= center
        xy[:, axis] = -xy[:, axis]
        xy += center
    xy = np.concatenate([xy, vis], axis=-1)
    return img, xy


def translate(img, xy, tx, ty):
    h, w = img.shape[:2]
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    img = cv2.warpAffine(img, M, (w, h))
    xy[:, 0] += tx/w
    xy[:, 1] += ty/h
    return img, xy


def warp_perspective(img, xy, rho):
    patch_size = 128
    top_point = (32,32)
    left_point = (patch_size+32, 32)
    bottom_point = (patch_size+32, patch_size+32)
    right_point = (32, patch_size+32)
    four_points = [top_point, left_point, bottom_point, right_point]
    h, w = img.shape[:2]

    perturbed_four_points = [
        (p[0] + np.random.uniform(-rho, rho), p[1] + np.random.uniform(-rho, rho))
        for p in four_points]

    M = cv2.getPerspectiveTransform(
        np.float32(four_points),
        np.float32(perturbed_four_points))

    warped_image = cv2.warpPerspective(img, M, (w, h))

    vis = xy[:, 2:]
    xy = xy[:, :2]
    xy *= [[w, h]]

    xyz = np.concatenate((xy, np.ones((xy.shape[0], 1))), axis=-1)
    xyz = np.matmul(M, xyz.T).T
    xy = xyz[:, :2] / xyz[:, 2:]

    xy /= [[w, h]]
    xy = np.concatenate([xy, vis], axis=-1)

    return warped_image, xy


def get_bounding_boxes(xy, size):
    xy[((xy[:, 0] - size / 2 <= 0) |
        (xy[:, 0] + size / 2 >= 1) |
        (xy[:, 1] - size / 2 <= 0) |
        (xy[:, 1] + size / 2 >= 1)), -1] = 0
    xywhc = []
    for i, _xy in enumerate(xy):
        if i < 4:
            cls = i + 1
        else:
            cls = 0
        if _xy[-1]:  # is visible
            xywhc.append([_xy[0], _xy[1], size, size, cls])
    xywhc = np.array(xywhc)
    return xywhc


def set_shapes(img, gt1, gt2, gt3, input_size):
    img.set_shape([input_size, input_size, 3])
    gt1.set_shape([input_size // 8, input_size // 8, 3, 10])
    gt2.set_shape([input_size // 16, input_size // 16, 3, 10])
    gt3.set_shape([input_size // 32, input_size // 32, 3, 10])
    return img, gt1, gt2, gt3


def set_shapes_tiny(img, gt1, gt2, input_size):
    img.set_shape([input_size, input_size, 3])
    gt1.set_shape([input_size // 16, input_size // 16, 3, 10])
    gt2.set_shape([input_size // 32, input_size // 32, 3, 10])
    return img, gt1, gt2


def load_tfds(
        cfg,
        bbox_to_gt_func,
        split='train',
        return_xy=False,
        batch_size=32,
        debug=False):

    data = get_splits(cfg.data.labels_path, cfg.data.dataset, split)
    img_path = osp.join(cfg.data.path, 'cropped_images', str(cfg.model.input_size))
    img_paths = [osp.join(img_path, folder, name) for (folder, name) in zip(data.img_folder, data.img_name)]

    xys = np.zeros((len(data), 7, 3))  # third column for visibility
    data.xy = data.xy.apply(np.array)
    for i, _xy in enumerate(data.xy):
        xys[i, :_xy.shape[0], :2] = _xy
        xys[i, :_xy.shape[0], 2] = 1
    xys = xys.astype(np.float32)

    if return_xy:
        dtypes = [tf.float32 for _ in range(2)]
    else:
        if cfg.model.tiny:
            dtypes = [tf.float32 for _ in range(3)]
        else:
            dtypes = [tf.float32 for _ in range(4)]

    AUTO = tf.data.experimental.AUTOTUNE if not debug else 1
    ds = tf.data.Dataset.from_tensor_slices((img_paths, xys))
    ds = ds.shuffle(10000).repeat()

    ds = ds.map(lambda path, xy:
                tf.py_function(
                    lambda path, xy: preprocess(path, xy, cfg, bbox_to_gt_func, split, return_xy),
                    [path, xy], dtypes),
                num_parallel_calls=AUTO)

    input_size = int(img_path.split('/')[-1])

    if not return_xy:
        if cfg.model.tiny:
            ds = ds.map(lambda img, gt1, gt2:
                        set_shapes_tiny(img, gt1, gt2, input_size),
                        num_parallel_calls=AUTO)
        else:
            ds = ds.map(lambda img, gt1, gt2, gt3:
                        set_shapes(img, gt1, gt2, gt3, input_size),
                        num_parallel_calls=AUTO)

    ds = ds.batch(batch_size).prefetch(AUTO)
    ds = data_generator(iter(ds), len(data), cfg.model.tiny) if not return_xy else ds
    return ds


class data_generator():
    """Wrap the tensorflow dataset in a generator so that we can combine
    gt into list because that's what the YOLOv4 loss function requires"""
    def __init__(self, tfds, n, tiny):
        self.tfds = tfds
        self.tiny = tiny
        self.n = n

    def __iter__(self):
        return self

    def __len__(self):
        return self.n

    def __next__(self):
        if self.tiny:
            img, gt1, gt2 = next(self.tfds)
            gt = [gt1, gt2]
        else:
            img, gt1, gt2, gt3 = next(self.tfds)
            gt = [gt1, gt2, gt3]
        return img, gt


if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.random.set_seed(0)
    np.random.seed(0)

    cfg = CN(new_allowed=True)
    cfg.merge_from_file('configs/aug_d2/tiny480_d2_20e_warp.yaml')

    from train import build_model

    yolo = build_model(cfg)

    yolo_dataset_object = yolo.load_dataset('dummy_dataset.txt', label_smoothing=0.)
    bbox_to_gt_func = yolo_dataset_object.bboxes_to_ground_truth

    ds = load_tfds(
        cfg,
        bbox_to_gt_func,
        split='train',
        return_xy=True,
        batch_size=1,
        debug=True)

    # for i, (img, (gt1, gt2)) in enumerate(ds):
    #     print(i, img.shape)
    #     print(gt1.shape, gt2.shape)
    #     img = (img.numpy()[0] * 255.).astype(np.uint8)[:, :, [2, 1, 0]]
    #     cv2.imshow('', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    for img, xy in ds:
        img = img[0].numpy()
        xy = xy[0].numpy()

        img = (img * 255.).astype(np.uint8)
        xy = xy[xy[:, -1] == 1, :2]
        img = draw(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), xy, cfg, False, True)

        cv2.imshow('', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
