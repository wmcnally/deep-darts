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
from predict import predict

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from yolov4.tf import YOLOv4
from yolov4.model import yolov4
from loss import YOLOv4Loss


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


def build_model(cfg, classes='classes'):
    yolo = YOLOv4(tiny=cfg.model.tiny)
    yolo.classes = classes
    yolo.input_size = (cfg.model.input_size, cfg.model.input_size)
    yolo.batch_size = cfg.train.batch_size
    make_model(yolo)
    return yolo


def train(cfg, strategy):
    img_path = osp.join(cfg.data.path, 'cropped_images', str(cfg.model.input_size))
    assert osp.exists(img_path), 'Could not find cropped images at {}'.format(img_path)

    tf.random.set_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)

    with strategy.scope():
        yolo = build_model(cfg)

    if cfg.model.weights_path:
        if cfg.model.weights_path.endswith('.h5'):
            yolo.model.load_weights(cfg.model.weights_path, by_name=True, skip_mismatch=True)
        else:
            if 'weights_layers' in cfg.model:
                pretrained_model = build_model(cfg).model
                pretrained_model.load_weights(cfg.model.weights_path)
                for module, pretrained_module in zip(yolo.model.layers, pretrained_model.layers):
                    for layer, pretrained_layer in zip(module.layers, pretrained_module.layers):
                        if layer.name in cfg.model.weights_layers:
                            layer.set_weights(pretrained_layer.get_weights())
                            print('Transferred pretrained weights to', layer.name)
                del pretrained_model
            else:
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
    n_val = val_ds.__len__()
    print('Train samples:', n_train)
    print('Val samples:', n_val)

    spe = int(np.ceil(n_train / (cfg.train.batch_size * strategy.num_replicas_in_sync)))

    with strategy.scope():
        lr = tf.keras.experimental.CosineDecay(cfg.train.lr, cfg.train.epochs * spe)
        optimizer = tf.keras.optimizers.Adam(lr)
        loss = YOLOv4Loss(
            batch_size=yolo.batch_size,
            iou_type=cfg.train.loss_type,
            verbose=cfg.train.loss_verbose)
        yolo.model.compile(optimizer=optimizer, loss=loss)

    val_steps = {'d1': 20, 'd2': 8}

    hist = yolo.model.fit(
        train_ds,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        verbose=cfg.train.verbose,
        validation_data=None if not cfg.train.val else val_ds,
        validation_steps=val_steps[cfg.data.dataset] // strategy.num_replicas_in_sync,
        steps_per_epoch=spe)

    yolo.save_weights(
        weights_path='./models/{}/weights'.format(cfg.model.name),
        weights_type=cfg.train.save_weights_type)

    pickle.dump(hist.history, open('./models/{}/history.pkl'.format(cfg.model.name), 'wb'))
    return yolo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='default')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    tpu, strategy = detect_hardware(tpu_name=None)
    yolo = train(cfg, strategy)
    predict(yolo, cfg, dataset=cfg.data.dataset, split='val')
