import os
import os.path as osp
import cv2
import pandas as pd
import numpy as np
from yacs.config import CfgNode as CN
import argparse

# used to convert dart angle to board number
BOARD_DICT = {
    0: '13', 1: '4', 2: '18', 3: '1', 4: '20', 5: '5', 6: '12', 7: '9', 8: '14', 9: '11',
    10: '8', 11: '16', 12: '7', 13: '19', 14: '3', 15: '17', 16: '2', 17: '15', 18: '10', 19: '6'
}


def crop_board(img_path, bbox=None, crop_info=(0, 0, 0), crop_pad=1.1):
    img = cv2.imread(img_path)
    if bbox is None:
        x, y, r = crop_info
        r = int(r * crop_pad)
        bbox = [y-r, y+r, x-r, x+r]
    crop = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    return crop, bbox


def on_click(event, x, y, flags, param):
    global xy, img_copy
    h, w = img_copy.shape[:2]
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(xy) < 7:
            xy.append([x/w, y/h])
            print_xy()
        else:
            print('Already annotated 7 points.')


def print_xy():
    global xy
    names = {
        0: 'cal_1', 1: 'cal_2', 2: 'cal_3', 3: 'cal_4',
        4: 'dart_1', 5: 'dart_2', 6: 'dart_3'}
    print('{}: {}'.format(names[len(xy)-1], xy[-1]))


def get_ellipses(xy, r_double=0.17, r_treble=0.1074):
    c = np.mean(xy[:4], axis=0)
    a1_double = ((xy[2][0] - xy[3][0]) ** 2 + (xy[2][1] - xy[3][1]) ** 2) ** 0.5 / 2
    a2_double = ((xy[0][0] - xy[1][0]) ** 2 + (xy[0][1] - xy[1][1]) ** 2) ** 0.5 / 2
    a1_treble = a1_double * (r_treble / r_double)
    a2_treble = a2_double * (r_treble / r_double)
    angle = np.arctan((xy[3, 1] - c[1]) / (xy[3, 0] - c[0])) / np.pi * 180
    return c, [a1_double, a2_double], [a1_treble, a2_treble], angle


def draw_ellipses(img, xy, num_pts=7):
    # img must be uint8
    xy = np.array(xy)
    if xy.shape[0] > num_pts:
        xy = xy.reshape((-1, 2))
    if np.mean(xy) < 1:
        h, w = img.shape[:2]
        xy[:, 0] *= w
        xy[:, 1] *= h
    c, a_double, a_treble, angle = get_ellipses(xy)
    angle = np.arctan((xy[3,1]-c[1])/(xy[3,0]-c[0]))/np.pi*180
    cv2.ellipse(img, (int(round(c[0])), int(round(c[1]))),
                (int(round(a_double[0])), int(round(a_double[1]))),
                int(round(angle)), 0, 360, (255, 255, 255))
    cv2.ellipse(img, (int(round(c[0])), int(round(c[1]))),
                (int(round(a_treble[0])), int(round(a_treble[1]))),
                int(round(angle)), 0, 360, (255, 255, 255))
    return img


def get_circle(xy):
    c = np.mean(xy[:4], axis=0)
    r = np.mean(np.linalg.norm(xy[:4] - c, axis=-1))
    return c, r


def board_radii(r_d, cfg):
    r_t = r_d * (cfg.board.r_treble / cfg.board.r_double)  # treble radius, in px
    r_ib = r_d * (cfg.board.r_inner_bull / cfg.board.r_double)  # inner bull radius, in px
    r_ob = r_d * (cfg.board.r_outer_bull / cfg.board.r_double) # outer bull radius, in px
    w_dt = cfg.board.w_double_treble * (r_d / cfg.board.r_double)  # width of double and treble
    return r_t, r_ob, r_ib, w_dt


def draw_circles(img, xy, cfg, color=(255, 255, 255)):
    c, r_d = get_circle(xy)  # double radius
    r_t, r_ob, r_ib, w_dt = board_radii(r_d, cfg)
    for r in [r_d, r_d - w_dt, r_t, r_t - w_dt, r_ib, r_ob]:
        cv2.circle(img, (round(c[0]), round(c[1])), round(r), color)
    return img


def transform(xy, img=None, angle=9, M=None):

    if xy.shape[-1] == 3:
        has_vis = True
        vis = xy[:, 2:]
        xy = xy[:, :2]
    else:
        has_vis = False

    if img is not None and np.mean(xy[:4]) < 1:
        h, w = img.shape[:2]
        xy *= [[w, h]]

    if M is None:
        c, r = get_circle(xy)  # not necessarily a circle
        # c is center of 4 calibration points, r is mean distance from center to calibration points

        src_pts = xy[:4].astype(np.float32)
        dst_pts = np.array([
            [c[0] - r * np.sin(np.deg2rad(angle)), c[1] - r * np.cos(np.deg2rad(angle))],
            [c[0] + r * np.sin(np.deg2rad(angle)), c[1] + r * np.cos(np.deg2rad(angle))],
            [c[0] - r * np.cos(np.deg2rad(angle)), c[1] + r * np.sin(np.deg2rad(angle))],
            [c[0] + r * np.cos(np.deg2rad(angle)), c[1] - r * np.sin(np.deg2rad(angle))]
        ]).astype(np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    xyz = np.concatenate((xy, np.ones((xy.shape[0], 1))), axis=-1).astype(np.float32)
    xyz_dst = np.matmul(M, xyz.T).T
    xy_dst = xyz_dst[:, :2] / xyz_dst[:, 2:]

    if img is not None:
        img = cv2.warpPerspective(img.copy(), M, (img.shape[1], img.shape[0]))
        xy_dst /= [[w, h]]

    if has_vis:
        xy_dst = np.concatenate([xy_dst, vis], axis=-1)

    return xy_dst, img, M


def get_dart_scores(xy, cfg, numeric=False):
    valid_cal_pts = xy[:4][(xy[:4, 0] > 0) & (xy[:4, 1] > 0)]
    if xy.shape[0] <= 4 or valid_cal_pts.shape[0] < 4:  # missing calibration point
        return []
    xy, _, _ = transform(xy.copy(), angle=0)
    c, r_d = get_circle(xy)
    r_t, r_ob, r_ib, w_dt = board_radii(r_d, cfg)
    xy -= c
    angles = np.arctan2(-xy[4:, 1], xy[4:, 0]) / np.pi * 180
    angles = [a + 360 if a < 0 else a for a in angles]  # map to 0-360
    distances = np.linalg.norm(xy[4:], axis=-1)
    scores = []
    for angle, dist in zip(angles, distances):
        if dist > r_d:
            scores.append('0')
        elif dist <= r_ib:
            scores.append('DB')
        elif dist <= r_ob:
            scores.append('B')
        else:
            number = BOARD_DICT[int(angle / 18)]
            if dist <= r_d and dist > r_d - w_dt:
                scores.append('D' + number)
            elif dist <= r_t and dist > r_t - w_dt:
                scores.append('T' + number)
            else:
                scores.append(number)
    if numeric:
        for i, s in enumerate(scores):
            if 'B' in s:
                if 'D' in s:
                    scores[i] = 50
                else:
                    scores[i] = 25
            else:
                if 'D' in s or 'T' in s:
                    scores[i] = int(s[1:])
                    scores[i] = scores[i] * 2 if 'D' in s else scores[i] * 3
                else:
                    scores[i] = int(s)
    return scores


def draw(img, xy, cfg, circles, score, color=(255, 255, 0)):
    xy = np.array(xy)
    if xy.shape[0] > 7:
        xy = xy.reshape((-1, 2))
    if np.mean(xy) < 1:
        h, w = img.shape[:2]
        xy[:, 0] *= w
        xy[:, 1] *= h
    if xy.shape[0] >= 4 and circles:
        img = draw_circles(img, xy, cfg)
    if xy.shape[0] > 4 and score:
        scores = get_dart_scores(xy, cfg)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    line_type = 1
    for i, [x, y] in enumerate(xy):
        if i < 4:
            c = (0, 255, 0)  # green
        else:
            c = color  # cyan
        x = int(round(x))
        y = int(round(y))
        if i >= 4:
            cv2.circle(img, (x, y), 1, c, 1)
            if score:
                txt = str(scores[i - 4])
            else:
                txt = str(i + 1)
            cv2.putText(img, txt, (x + 8, y), font,
                    font_scale, c, line_type)
        else:
            cv2.circle(img, (x, y), 1, c, 1)
            cv2.putText(img, str(i + 1), (x + 8, y), font,
                        font_scale, c, line_type)
    return img


def adjust_xy(idx):
    global xy, img_copy
    key = cv2.waitKey(0) & 0xFF
    xy = np.array(xy)
    h, w = img_copy.shape[:2]
    xy[:, 0] *= w; xy[:, 1] *= h
    if key == 52:  # one pixel left
        if idx == -1:
            xy[:, 0] -= 1
        else:
            xy[idx, 0] -= 1
    if key == 56:  # one pixel up
        if idx == -1:
            xy[:, 1] -= 1
        else:
            xy[idx, 1] -= 1
    if key == 54:  # one pixel right
        if idx == -1:
            xy[:, 0] += 1
        else:
            xy[idx, 0] += 1
    if key == 50:  # one pixel down
        if idx == -1:
            xy[:, 1] += 1
        else:
            xy[idx, 1] += 1
    xy[:, 0] /= w; xy[:, 1] /= h
    xy = xy.tolist()


def add_last_dart(annot, data_path, folder):
    csv_path = osp.join(data_path, 'annotations', folder + '.csv')
    if osp.isfile(csv_path):
        dart_labels = []
        csv = pd.read_csv(csv_path)
        for idx in csv.index.values:
            for c in csv.columns:
                dart_labels.append(str(csv.loc[idx, c]))
        annot['last_dart'] = dart_labels
    return annot


def get_bounding_box(img_path, scale=0.2):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, None, fx=scale, fy=scale)
    h, w = img_resized.shape[:2]
    xy_bbox = []

    def on_click_bbox(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(xy_bbox) < 2:
                xy_bbox.append([
                    round((x / w) * img.shape[1]),
                    round((y / h) * img.shape[0])])

    window = 'get bbox'
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click_bbox)
    while len(xy_bbox) < 2:
        # print(xy_bbox)
        cv2.imshow(window, img_resized)
        key = cv2.waitKey(100)
        if key == ord('q'):  # quit
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    assert len(xy_bbox) == 2, 'click 2 points to get bounding box'
    xy_bbox = np.array(xy_bbox)
    # bbox = [y1 y2 x1 x2]
    bbox = [min(xy_bbox[:, 1]), max(xy_bbox[:, 1]), min(xy_bbox[:, 0]), max(xy_bbox[:, 0])]
    return bbox


def main(cfg, folder, scale, draw_circles, dart_score=True):
    global xy, img_copy
    img_dir = osp.join(cfg.data.path, 'images', folder)
    imgs = sorted(os.listdir(img_dir))
    annot_path = osp.join(cfg.data.path, 'annotations', folder + '.pkl')
    if osp.isfile(annot_path):
        annot = pd.read_pickle(annot_path)
    else:
        annot = pd.DataFrame(columns=['img_name', 'bbox', 'xy'])
        annot['img_name'] = imgs
        annot['bbox'] = None
        annot['xy'] = None
        annot = add_last_dart(annot, cfg.data.path, folder)

    i = 0
    for j in range(len(annot)):
        a = annot.iloc[j,:]
        if a['bbox'] is not None:
            i = j

    while i < len(imgs):
        xy = []
        a = annot.iloc[i,:]
        print('Annotating {}'.format(a['img_name']))
        if a['bbox'] is None:
            if i == 0:
                bbox = get_bounding_box(osp.join(img_dir, a['img_name']))
            if i > 0:
                last_a = annot.iloc[i-1,:]
                if last_a['xy'] is not None:
                    xy = last_a['xy'].copy()
            else:
                xy = []
        else:
            bbox, xy = a['bbox'], a['xy']

        crop, _ = crop_board(osp.join(img_dir, a['img_name']), bbox=bbox)
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
        cv2.putText(crop, '{}/{} {}'.format(i+1, len(annot), a['img_name']), (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        img_copy = crop.copy()

        cv2.namedWindow(folder)
        cv2.setMouseCallback(folder, on_click)
        while True:
            img_copy = draw(img_copy, xy, cfg, draw_circles, dart_score)
            cv2.imshow(folder, img_copy)
            key = cv2.waitKey(100) & 0xFF  # update every 100 ms

            if key == ord('q'):  # quit
                cv2.destroyAllWindows()
                i = len(imgs)
                break

            if key == ord('b'):  # draw new bounding box
                idx = annot[(annot['img_name'] == a['img_name'])].index.values[0]
                annot.at[idx, 'bbox'] = get_bounding_box(osp.join(img_dir, a['img_name']), scale)
                break

            if key == ord('.'):
                i += 1
                img_copy = crop.copy()
                break

            if key == ord(','):
                if i > 0:
                    i += -1
                    img_copy = crop.copy()
                    break

            if key == ord('z'):  # undo keypoint
                xy = xy[:-1]
                img_copy = crop.copy()

            if key == ord('x'):  # reset annotation
                idx = annot[(annot['img_name'] == a['img_name'])].index.values[0]
                annot.at[idx, 'xy'] = None,
                annot.at[idx, 'bbox'] = None
                annot.to_pickle(annot_path)
                break

            if key == ord('d'):  # delete img
                print('Are you sure you want to delete this image? (y/n)')
                key = cv2.waitKey(0) & 0xFF
                if key == ord('y'):
                    idx = annot[(annot['img_name'] == a['img_name'])].index.values[0]
                    annot = annot.drop([idx])
                    annot.to_pickle(annot_path)
                    os.remove(osp.join(img_dir, a['img_name']))
                    print('Deleted image {}'.format(a['img_name']))
                    break
                else:
                    print('Image not deleted.')
                    continue

            if key == ord('a'):  # accept keypoints
                idx = annot[(annot['img_name'] == a['img_name'])].index.values[0]
                annot.at[idx, 'xy'] = xy
                annot.at[idx, 'bbox'] = bbox
                annot.to_pickle(annot_path)
                i += 1
                break

            if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('0')]:
                adjust_xy(idx=key - 49)  # ord('1') = 49
                img_copy = crop.copy()
                continue


if __name__ == '__main__':
    import sys
    sys.path.append('../../')
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--img-folder', default='d2_04_05_2020')
    parser.add_argument('-s', '--scale', type=float, default=0.5)
    parser.add_argument('-d', '--draw-circles', action='store_true')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file('../configs/tiny480_20e.yaml')

    main(cfg, args.img_folder, args.scale, args.draw_circles)
