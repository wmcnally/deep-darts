import os
import os.path as osp
import cv2


def crop_board(img_path, bbox=None, crop_info=(0, 0, 0), crop_pad=1.1):
    img = cv2.imread(img_path)
    # print('original image shape: {}'.format(img.shape))
    if bbox is None:
        x, y, r = crop_info
        r = int(r * crop_pad)
        bbox = [y-r, y+r, x-r, x+r]
    crop = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    return crop, bbox


if __name__ == '__main__':
    CROP_INFO = [2300, 1500, 800]
    folder = 'debug'
    from yacs.config import CfgNode as CN
    cfg = CN(new_allowed=True)
    cfg.merge_from_file('../configs/default.yaml')
    imgs = sorted(os.listdir(osp.join(cfg.data.path, 'images', folder)))
    for img_name in imgs:
        img_path = osp.join(cfg.data.path, 'images', folder, img_name)
        crop, bbox = crop_board(img_path, crop_info=CROP_INFO)
        print(bbox)
        cv2.imshow('', crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
