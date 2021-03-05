from os import makedirs, path
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.losses import BinaryCrossentropy, Loss, Reduction


class YOLOv4Loss(Loss):
    """
    Patched version of loss to fix potential nan with conf_loss
    """

    def __init__(self, batch_size, iou_type, verbose=0):
        super(YOLOv4Loss, self).__init__(name="YOLOv4Loss")
        self.batch_size = batch_size
        if iou_type == "iou":
            self.bbox_xiou = bbox_iou
        elif iou_type == "giou":
            self.bbox_xiou = bbox_giou
        elif iou_type == "ciou":
            self.bbox_xiou = bbox_ciou

        self.verbose = verbose

        self.while_cond = lambda i, iou: tf.less(i, self.batch_size)

        self.prob_binaryCrossentropy = BinaryCrossentropy(
            reduction=Reduction.NONE
        )

    def call(self, y_true, y_pred):
        """
        @param `y_true`: Dim(batch, g_height, g_width, 3,
                                (b_x, b_y, b_w, b_h, conf, prob_0, prob_1, ...))
        @param `y_pred`: Dim(batch, g_height, g_width, 3,
                                (b_x, b_y, b_w, b_h, conf, prob_0, prob_1, ...))
        """
        if len(y_pred.shape) == 4:
            _, g_height, g_width, box_size = y_pred.shape
            box_size = box_size // 3
        else:
            _, g_height, g_width, _, box_size = y_pred.shape

        y_true = tf.reshape(
            y_true, shape=(-1, g_height * g_width * 3, box_size)
        )
        y_pred = tf.reshape(
            y_pred, shape=(-1, g_height * g_width * 3, box_size)
        )

        truth_xywh = y_true[..., 0:4]
        truth_conf = y_true[..., 4:5]
        truth_prob = y_true[..., 5:]

        num_classes = truth_prob.shape[-1]

        pred_xywh = y_pred[..., 0:4]
        pred_conf = y_pred[..., 4:5]
        pred_prob = y_pred[..., 5:]

        one_obj = truth_conf
        num_obj = tf.reduce_sum(one_obj, axis=[1, 2])
        one_noobj = 1.0 - one_obj
        # Dim(batch, g_height * g_width * 3, 1)
        one_obj_mask = one_obj > 0.5

        zero = tf.zeros((1, g_height * g_width * 3, 1), dtype=tf.float32)

        # IoU Loss
        xiou = self.bbox_xiou(truth_xywh, pred_xywh)
        xiou_scale = 2.0 - truth_xywh[..., 2:3] * truth_xywh[..., 3:4]
        xiou_loss = one_obj * xiou_scale * (1.0 - xiou[..., tf.newaxis])
        xiou_loss = 3 * tf.reduce_mean(tf.reduce_sum(xiou_loss, axis=(1, 2)))

        # Confidence Loss
        i0 = tf.constant(0)

        def body(i, max_iou):
            object_mask = tf.reshape(one_obj_mask[i, ...], shape=(-1,))
            truth_bbox = tf.boolean_mask(truth_xywh[i, ...], mask=object_mask)
            # g_height * g_width * 3,      1, xywh
            #               1, answer, xywh
            #   => g_height * g_width * 3, answer
            _max_iou0 = tf.cond(
                tf.equal(num_obj[i], 0),
                lambda: zero,
                lambda: tf.reshape(
                    tf.reduce_max(
                        bbox_iou(
                            pred_xywh[i, :, tf.newaxis, :],
                            truth_bbox[tf.newaxis, ...],
                        ),
                        axis=-1,
                    ),
                    shape=(1, -1, 1),
                ),
            )
            # 1, g_height * g_width * 3, 1
            _max_iou1 = tf.cond(
                tf.equal(i, 0),
                lambda: _max_iou0,
                lambda: tf.concat([max_iou, _max_iou0], axis=0),
            )
            return tf.add(i, 1), _max_iou1

        _, max_iou = tf.while_loop(
            self.while_cond,
            body,
            [i0, zero],
            shape_invariants=[
                i0.get_shape(),
                tf.TensorShape([None, g_height * g_width * 3, 1]),
            ],
        )

        conf_obj_loss = one_obj * (0.0 - backend.log(pred_conf + backend.epsilon()))  # changed eps from 1e-9
        conf_noobj_loss = (
            one_noobj
            * tf.cast(max_iou < 0.5, dtype=tf.float32)
            * (0.0 - backend.log(1.0 - pred_conf + backend.epsilon()))  # changed eps from 1e-9
        )
        conf_loss = tf.reduce_mean(
            tf.reduce_sum(conf_obj_loss + conf_noobj_loss, axis=(1, 2))
        )

        # Probabilities Loss
        prob_loss = self.prob_binaryCrossentropy(truth_prob, pred_prob)
        prob_loss = one_obj * prob_loss[..., tf.newaxis]
        prob_loss = tf.reduce_mean(
            tf.reduce_sum(prob_loss, axis=(1, 2)) * num_classes
        )

        total_loss = xiou_loss + conf_loss + prob_loss

        if self.verbose != 0:
            # tf.print(
            #     f"grid: {g_height}*{g_width}, iou_loss: {xiou_loss:7.3f}, conf_loss: {conf_loss:7.3f}, prob_loss: {prob_loss:7.3f}, total_loss: {total_loss:7.3f}"
            # )
            tf.print(
                f"grid: {g_height}*{g_width}",
                "iou_loss:",
                xiou_loss,
                "conf_loss:",
                conf_loss,
                "prob_loss:",
                prob_loss,
                "total_loss",
                total_loss,
            )

        return total_loss


def bbox_iou(bboxes1, bboxes2):
    """
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = inter_area / (union_area + 1e-8)

    return iou


def bbox_giou(bboxes1, bboxes2):
    """
    Generalized IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = inter_area / (union_area + 1e-8)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-8)

    return giou


def bbox_ciou(bboxes1, bboxes2):
    """
    Complete IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = inter_area / (union_area + 1e-8)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up

    c_2 = enclose_section[..., 0] ** 2 + enclose_section[..., 1] ** 2

    center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]

    rho_2 = center_diagonal[..., 0] ** 2 + center_diagonal[..., 1] ** 2

    diou = iou - rho_2 / (c_2 + 1e-8)

    v = (
        (
            tf.math.atan(bboxes1[..., 2] / (bboxes1[..., 3] + 1e-8))
            - tf.math.atan(bboxes2[..., 2] / (bboxes2[..., 3] + 1e-8))
        )
        * 2
        / 3.1415926536
    ) ** 2

    alpha = v / (1 - iou + v + 1e-8)

    ciou = diou - alpha * v

    return ciou
