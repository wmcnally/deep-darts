import os
import os.path as osp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from yacs.config import CfgNode as CN
from dataloader import load_tfds
import numpy as np
import argparse
from utils import detect_hardware
import pickle
from tensorflow.keras import layers
import random

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from yolov4.tf import YOLOv4
from yolov4.model import yolov4


def make_model(
        yolo,
        activation0: str = "mish",
        activation1: str = "leaky",
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
):
    """Use this function instead of yolo.make_model()"""
    yolo._has_weights = False
    # height, width, channels
    inputs = layers.Input([yolo.input_size[1], yolo.input_size[0], 3])
    if yolo.tiny:
        yolo.model = yolov4.YOLOv4Tiny(
            anchors=yolo.anchors,
            num_classes=len(yolo.classes),
            xyscales=yolo.xyscales,
            activation=activation1,
            kernel_regularizer=kernel_regularizer,
        )
    else:
        yolo.model = yolov4.YOLOv4(
            anchors=yolo.anchors,
            num_classes=len(yolo.classes),
            xyscales=yolo.xyscales,
            activation0=activation0,
            activation1=activation1,
            kernel_regularizer=kernel_regularizer,
        )
    yolo.model(inputs)


def build_model(cfg):
    yolo = YOLOv4(tiny=cfg.model.tiny)
    yolo.classes = 'classes'
    yolo.input_size = (cfg.model.input_size, cfg.model.input_size)
    yolo.batch_size = cfg.train.batch_size
    # yolo.make_model()
    make_model(yolo)
    return yolo


def train(cfg, strategy):
    img_path = osp.join(cfg.data.path, 'cropped_images', str(cfg.model.input_size))
    assert osp.exists(img_path)

    tf.random.set_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)

    with strategy.scope():
        yolo = build_model(cfg)

    if cfg.model.weights_path:
        yolo.load_weights(
            weights_path=cfg.model.weights_path,
            weights_type=cfg.model.weights_type)

    yolo_dataset_object = yolo.load_dataset('dummy_dataset.txt', label_smoothing=0.)
    bbox_to_gt_func = yolo_dataset_object.bboxes_to_ground_truth

    train_ds = load_tfds(
        cfg,
        bbox_to_gt_func,
        split='train',
        batch_size=cfg.train.batch_size * strategy.num_replicas_in_sync)

    val_ds = load_tfds(
        cfg,
        bbox_to_gt_func,
        split='val',
        batch_size=cfg.train.batch_size * strategy.num_replicas_in_sync)

    n_train = train_ds.__len__()
    spe = int(np.ceil(n_train / (cfg.train.batch_size * strategy.num_replicas_in_sync)))

    with strategy.scope():
        lr = tf.keras.experimental.CosineDecay(cfg.train.lr, cfg.train.epochs * spe)
        optimizer = tf.keras.optimizers.Adam(lr)
        yolo.compile(optimizer=optimizer, loss_iou_type="ciou", loss_verbose=0)

    hist = yolo.model.fit(
        train_ds,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        verbose=cfg.train.verbose,
        validation_data=val_ds,
        validation_steps=20 // strategy.num_replicas_in_sync,
        steps_per_epoch=spe)

    yolo.save_weights(
        weights_path='./models/{}/weights'.format(cfg.model.name),
        weights_type=cfg.train.save_weights_type)

    pickle.dump(hist.history, open('./models/{}/history.pkl'.format(cfg.model.name), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='default')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    tpu, strategy = detect_hardware(tpu_name=None)
    train(cfg, strategy)
