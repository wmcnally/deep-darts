import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os.path as osp
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from dataset.annotate import draw
from yacs.config import CfgNode as CN
from yolov4.tf import YOLOv4
from yolov4.tf.dataset import cut_out


d1_val = ['02_06_2020', '02_16_2020', '02_22_2020']
d1_test = ['03_03_2020', '03_19_2020', '03_23_2020', '03_27_2020', '03_28_2020', '03_30_2020', '03_31_2020']

d2_val = ['d2_03_03_2020']
d2_test = ['d2_03_08_2020']


def get_splits(path='./dataset/labels.pkl', dataset='d1', split='train'):
    assert dataset in ['d1', 'd2'], "dataset must be in ['d1', 'd2']"
    assert split in [None, 'train', 'val', 'test'], "split must be in [None, 'train', 'val', 'test']"
    df = pd.read_pickle(path)
    splits = {}
    if dataset == 'd1':
        df = df[np.logical_not(df.img_folder.str.contains('d2'))]
        splits['val'] = df[np.isin(df.img_folder, d1_val)]
        splits['test'] = df[np.isin(df.img_folder, d1_test)]
        splits['train'] = df[np.logical_not(np.isin(df.img_folder, d1_val + d1_test))]
    else:
        df = df[df.img_folder.str.contains('d2')]
        splits['val'] = df[np.isin(df.img_folder, d2_val)]
        splits['test'] = df[np.isin(df.img_folder, d2_test)]
        splits['train'] = df[np.logical_not(np.isin(df.img_folder, d2_val + d2_test))]
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

    if split == 'train':
        # augmentation
        if cfg.aug.flip_lr_prob or cfg.aug.flip_ud_prob:
            img, xy = align_board(img, xy)

        if cfg.aug.flip_lr_prob and np.random.uniform() < cfg.aug.flip_lr_prob:
            img, xy = flip(img, xy, direction='lr')

        if cfg.aug.flip_ud_prob and np.random.uniform() < cfg.aug.flip_ud_prob:
            img, xy = flip(img, xy, direction='ud')

        if cfg.aug.rot_prob and np.random.uniform() < cfg.aug.rot_prob:
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


# def mixup(image, label, PROBABILITY=1.0):
#     """Modified from https://www.kaggle.com/cdeotte/cutmix-and-mixup-on-gpu-tpu"""
#     # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
#     # output - a batch of images with mixup applied
#     DIM = IMAGE_SIZE[0]
#     CLASSES = 104
#
#     imgs = [];
#     labs = []
#     for j in range(AUG_BATCH):
#         # DO MIXUP WITH PROBABILITY DEFINED ABOVE
#         P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.float32)
#         # CHOOSE RANDOM
#         k = tf.cast(tf.random.uniform([], 0, AUG_BATCH), tf.int32)
#         a = tf.random.uniform([], 0, 1) * P  # this is beta dist with alpha=1.0
#         # MAKE MIXUP IMAGE
#         img1 = image[j,]
#         img2 = image[k,]
#         imgs.append((1 - a) * img1 + a * img2)
#         # MAKE CUTMIX LABEL
#         if len(label.shape) == 1:
#             lab1 = tf.one_hot(label[j], CLASSES)
#             lab2 = tf.one_hot(label[k], CLASSES)
#         else:
#             lab1 = label[j,]
#             lab2 = label[k,]
#         labs.append((1 - a) * lab1 + a * lab2)
#
#     # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
#     image2 = tf.reshape(tf.stack(imgs), (AUG_BATCH, DIM, DIM, 3))
#     label2 = tf.reshape(tf.stack(labs), (AUG_BATCH, CLASSES))
#     return image2, label2


def get_bounding_boxes(xy, size, e=1e-5):
    xy = xy[(
        (xy[:, -1] == 1) &
        (xy[:, 0] - size / 2 > 0 + e) &
        (xy[:, 0] + size / 2 < 1 - e) &
        (xy[:, 1] - size / 2 > 0 + e) &
        (xy[:, 1] + size / 2 < 1 - e)), :2]
    # xywhc same format as yolov4 code: [center_x, center_y, w, h, class_id]
    xywhc = np.zeros((xy.shape[0], 5))
    xywhc[:, :2] = xy
    xywhc[:, 2:4] = size
    xywhc[:4, -1] = list(range(1, 5))  # classes 1 through 4 for calibration points, dart is class 0
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
        batch_size=32):

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

    AUTO = tf.data.experimental.AUTOTUNE
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

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.random.set_seed(0)
    np.random.seed(0)

    cfg = CN(new_allowed=True)
    cfg.merge_from_file('configs/debug.yaml')

    from train import build_model

    yolo = build_model(cfg)

    yolo_dataset_object = yolo.load_dataset('dummy_dataset.txt', label_smoothing=0.)
    bbox_to_gt_func = yolo_dataset_object.bboxes_to_ground_truth

    ds = load_tfds(
        cfg,
        bbox_to_gt_func,
        split='train',
        return_xy=True,
        batch_size=1)

    # for i, (img, (gt1, gt2)) in enumerate(ds):
    #     print(i, img.shape)
    #     img = (img.numpy()[0] * 255.).astype(np.uint8)[:, :, [2, 1, 0]]
    #     cv2.imshow('', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    for img, xy in ds:
        img = img[0]
        xy = xy[0]

        img = (img.numpy() * 255.).astype(np.uint8)[:, :, [2, 1, 0]]
        xy = xy.numpy()
        xy = xy[xy[:, -1] == 1, :2]
        img = draw(img.copy(), xy, cfg, False, True)

        cv2.imshow('', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
