# -*- coding: utf-8 -*-
# @Time    : 2019/1/12 13:06

import cv2
import numbers
import math
import random
import numpy as np
from skimage.util import random_noise


def show_pic(img, bboxes=None, name='pic'):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    show_img = img.copy()
    if not isinstance(bboxes, np.ndarray):
        bboxes = np.array(bboxes)
    for point in bboxes.astype(np.int):
        cv2.line(show_img, tuple(point[0]), tuple(point[1]), (255, 0, 0), 2)
        cv2.line(show_img, tuple(point[1]), tuple(point[2]), (255, 0, 0), 2)
        cv2.line(show_img, tuple(point[2]), tuple(point[3]), (255, 0, 0), 2)
        cv2.line(show_img, tuple(point[3]), tuple(point[0]), (255, 0, 0), 2)
    # cv2.namedWindow(name, 0)  # 1表示原图
    # cv2.moveWindow(name, 0, 0)
    # cv2.resizeWindow(name, 1200, 800)  # 可视化的图片大小
    cv2.imshow(name, show_img)


# 图像均为cv2读取
class DataAugment():
    def __init__(self):
        pass

    def add_noise(self, im: np.ndarray):
        """
        对图片加噪声
        :param img: 图像array
        :return: 加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        """
        return (random_noise(im, mode='gaussian', clip=True) * 255).astype(im.dtype)

    def random_scale(self, im: np.ndarray, mask: np.ndarray, scales: np.ndarray or list) -> tuple:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param im: 原图
        :param text_polys: 文本框
        :param scales: 尺度
        :return: 经过缩放的图片和文本
        """
#         tmp_text_polys = text_polys.copy()
        rd_scale = float(np.random.choice(scales))
        
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        mask = cv2.resize(mask, dsize=None, fx=rd_scale, fy=rd_scale)
        return im, mask


    def random_rotate_img_bbox(self, img, text_polys, degrees: numbers.Number or list or tuple or np.ndarray,
                               same_size=False):
        """
        从给定的角度中选择一个角度，对图片和文本框进行旋转
        :param img: 图片
        :param text_polys: 文本框
        :param degrees: 角度，可以是一个数值或者list
        :param same_size: 是否保持和原图一样大
        :return: 旋转后的图片和角度
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            degrees = (-degrees, degrees)
        elif isinstance(degrees, list) or isinstance(degrees, tuple) or isinstance(degrees, np.ndarray):
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            degrees = degrees
        else:
            raise Exception('degrees must in Number or list or tuple or np.ndarray')
        # ---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        angle = np.random.uniform(degrees[0], degrees[1])

        if same_size:
            nw = w
            nh = h
        else:
            # 角度变弧度
            rangle = np.deg2rad(angle)
            # 计算旋转之后图像的w, h
            nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
            nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
        # 构造仿射矩阵
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        # 计算原图中心点到新图中心点的偏移量
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # 更新仿射矩阵
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_text_polys = list()
        for bbox in text_polys:
            point1 = np.dot(rot_mat, np.array([bbox[0, 0], bbox[0, 1], 1]))
            point2 = np.dot(rot_mat, np.array([bbox[1, 0], bbox[1, 1], 1]))
            point3 = np.dot(rot_mat, np.array([bbox[2, 0], bbox[2, 1], 1]))
            point4 = np.dot(rot_mat, np.array([bbox[3, 0], bbox[3, 1], 1]))
            rot_text_polys.append([point1, point2, point3, point4])
        return rot_img, np.array(rot_text_polys, dtype=np.float32)

    def random_crop_img_bboxes(self, im: np.ndarray, text_polys: np.ndarray, max_tries=50) -> tuple:
        """
        从图片中裁剪出 cropsize大小的图片和对应区域的文本框
        :param im: 图片
        :param text_polys: 文本框
        :param max_tries: 最大尝试次数
        :return: 裁剪后的图片和文本框
        """
        h, w, _ = im.shape
        pad_h = h // 10
        pad_w = w // 10
        h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
        w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
        for poly in text_polys:
            poly = np.round(poly, decimals=0).astype(np.int32)  # 四舍五入取整
            minx = np.min(poly[:, 0])
            maxx = np.max(poly[:, 0])
            w_array[minx + pad_w:maxx + pad_w] = 1  # 将文本区域的在w_array上设为1，表示x轴方向上这部分位置有文本
            miny = np.min(poly[:, 1])
            maxy = np.max(poly[:, 1])
            h_array[miny + pad_h:maxy + pad_h] = 1  # 将文本区域的在h_array上设为1，表示y轴方向上这部分位置有文本
        # 在两个轴上 拿出背景位置去进行随机的位置选择，避免选择的区域穿过文本
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        if len(h_axis) == 0 or len(w_axis) == 0:
            # 整张图全是文本的情况下，直接返回
            return im, text_polys
        for i in range(max_tries):
            xx = np.random.choice(w_axis, size=2)
            # 对选择区域进行边界控制
            xmin = np.min(xx) - pad_w
            xmax = np.max(xx) - pad_w
            xmin = np.clip(xmin, 0, w - 1)
            xmax = np.clip(xmax, 0, w - 1)
            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy) - pad_h
            ymax = np.max(yy) - pad_h
            ymin = np.clip(ymin, 0, h - 1)
            ymax = np.clip(ymax, 0, h - 1)
            if xmax - xmin < 0.1 * w or ymax - ymin < 0.1 * h:
                # 选择的区域过小
                # area too small
                continue
            if text_polys.shape[0] != 0:  # 这个判断不知道干啥的
                poly_axis_in_area = (text_polys[:, :, 0] >= xmin) & (text_polys[:, :, 0] <= xmax) \
                                    & (text_polys[:, :, 1] >= ymin) & (text_polys[:, :, 1] <= ymax)
                selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
            else:
                selected_polys = []
            if len(selected_polys) == 0:
                # 区域内没有文本
                continue
            im = im[ymin:ymax + 1, xmin:xmax + 1, :]
            polys = text_polys[selected_polys]
            # 坐标调整到裁剪图片上
            polys[:, :, 0] -= xmin
            polys[:, :, 1] -= ymin
            return im, polys
        return im, text_polys

    def random_crop_image_pse(self, im: np.ndarray, text_polys: np.ndarray, input_size) -> tuple:
        """
        从图片中裁剪出 cropsize大小的图片和对应区域的文本框
        :param im: 图片
        :param text_polys: 文本框
        :param input_size: 输出图像大小
        :return: 裁剪后的图片和文本框
        """
        h, w, _ = im.shape
        short_edge = min(h, w)
        if short_edge < input_size:
            # 保证短边 >= inputsize
            scale = input_size / short_edge
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            text_polys *= scale
            h, w, _ = im.shape
        # 计算随机范围
        w_range = w - input_size
        h_range = h - input_size
        for _ in range(50):
            xmin = random.randint(0, w_range)
            ymin = random.randint(0, h_range)
            xmax = xmin + input_size
            ymax = ymin + input_size
            if text_polys.shape[0] != 0:
                selected_polys = []
                for poly in text_polys:
                    if poly[:, 0].max() < xmin or poly[:, 0].min() > xmax or \
                            poly[:, 1].max() < ymin or poly[:, 1].min() > ymax:
                        continue
                    # area_p = cv2.contourArea(poly)
                    poly[:, 0] -= xmin
                    poly[:, 1] -= ymin
                    poly[:, 0] = np.clip(poly[:, 0], 0, input_size)
                    poly[:, 1] = np.clip(poly[:, 1], 0, input_size)
                    # rect = cv2.minAreaRect(poly)
                    # area_n = cv2.contourArea(poly)
                    # h1, w1 = rect[1]
                    # if w1 < 10 or h1 < 10 or area_n / area_p < 0.5:
                    #     continue
                    selected_polys.append(poly)
            else:
                selected_polys = []
            # if len(selected_polys) == 0:
                # 区域内没有文本
                # continue
            im = im[ymin:ymax, xmin:xmax, :]
            polys = np.array(selected_polys)
            return im, polys
        return im, text_polys

    def resize_author(self,imgs, img_size):
        h, w = imgs[0].shape[0:2]
        th, tw = img_size
        if w == tw and h == th:
            return imgs

        # return i, j, th, tw
        for idx in range(len(imgs)):
            imgs[idx] = cv2.resize(imgs[idx], img_size)
        return imgs

    def random_horizontal_flip(self,imgs):
        if random.random() < 0.5:
            for i in range(len(imgs)):
                imgs[i] = np.flip(imgs[i], axis=1).copy()
        return imgs

    def random_rotate(self,imgs):
        max_angle = 10
        angle = random.random() * 2 * max_angle - max_angle
        for i in range(len(imgs)):
            img = imgs[i]
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] = img_rotation
        return imgs

    def random_crop_padding(self,imgs, target_size):
        """ using padding and the final crop size is (800, 800) """
        h, w = imgs[0].shape[0:2]
        t_w, t_h = target_size
        p_w, p_h = target_size
        if w == t_w and h == t_h:
            return imgs

        t_h = t_h if t_h < h else h
        t_w = t_w if t_w < w else w

        if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
            # make sure to crop the text region
            tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
            tl[tl < 0] = 0
            br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
            br[br < 0] = 0
            br[0] = min(br[0], h - t_h)
            br[1] = min(br[1], w - t_w)

            i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
            j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
        else:
            i = random.randint(0, h - t_h) if h - t_h > 0 else 0
            j = random.randint(0, w - t_w) if w - t_w > 0 else 0

        n_imgs = []
        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                s3_length = int(imgs[idx].shape[-1])
                img = imgs[idx][i:i + t_h, j:j + t_w, :]
                img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                           value=tuple(0 for i in range(s3_length)))
            else:
                img = imgs[idx][i:i + t_h, j:j + t_w]
                img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
            n_imgs.append(img_p)
        return n_imgs

    def random_crop_author(self,imgs, img_size):
        h, w = imgs[0].shape[0:2]
        th, tw = img_size
        if w == tw and h == th:
            return imgs

        # label中存在文本实例，并且按照概率进行裁剪
        if np.max(imgs[1][:,:]) > 0 and random.random() > 3.0 / 8.0:
            # 文本实例的top left点
            tl = np.min(np.where(imgs[1][:,:] > 0), axis=1) - img_size
            tl[tl < 0] = 0
            # 文本实例的 bottom right 点
            br = np.max(np.where(imgs[1][:,:] > 0), axis=1) - img_size
            br[br < 0] = 0
            # 保证选到右下角点是，有足够的距离进行crop
            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)
            for _ in range(50000):
                i = random.randint(tl[0], br[0])
                j = random.randint(tl[1], br[1])
                # 保证最小的图有文本
                if imgs[1][:,:][i:i + th, j:j + tw].sum() <= 0:
                    continue
                else:
                    break
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

        # return i, j, th, tw
        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
            else:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw]
        return imgs

    def resize(self, im: np.ndarray, text_polys: np.ndarray,
               input_size: numbers.Number or list or tuple or np.ndarray, keep_ratio: bool = False) -> tuple:
        """
        对图片和文本框进行resize
        :param im: 图片
        :param text_polys: 文本框
        :param input_size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :param keep_ratio: 是否保持长宽比
        :return: resize后的图片和文本框
        """
        if isinstance(input_size, numbers.Number):
            if input_size < 0:
                raise ValueError("If input_size is a single number, it must be positive.")
            input_size = (input_size, input_size)
        elif isinstance(input_size, list) or isinstance(input_size, tuple) or isinstance(input_size, np.ndarray):
            if len(input_size) != 2:
                raise ValueError("If input_size is a sequence, it must be of len 2.")
            input_size = (input_size[0], input_size[1])
        else:
            raise Exception('input_size must in Number or list or tuple or np.ndarray')
        if keep_ratio:
            # 将图片短边pad到和长边一样
            h, w, c = im.shape
            max_h = max(h, input_size[0])
            max_w = max(w, input_size[1])
            im_padded = np.zeros((max_h, max_w, c), dtype=np.uint8)
            im_padded[:h, :w] = im.copy()
            im = im_padded
        text_polys = text_polys.astype(np.float32)
        h, w, _ = im.shape
        im = cv2.resize(im, input_size)
        w_scale = input_size[0] / float(w)
        h_scale = input_size[1] / float(h)
        text_polys[:, :, 0] *= w_scale
        text_polys[:, :, 1] *= h_scale
        return im, text_polys

    def horizontal_flip(self, im: np.ndarray, text_polys: np.ndarray) -> tuple:
        """
        对图片和文本框进行水平翻转
        :param im: 图片
        :param text_polys: 文本框
        :return: 水平翻转之后的图片和文本框
        """
        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 1)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 0] = w - flip_text_polys[:, :, 0]
        return flip_im, flip_text_polys

    def vertical_flip(self, im: np.ndarray, text_polys: np.ndarray) -> tuple:
        """
         对图片和文本框进行竖直翻转
        :param im: 图片
        :param text_polys: 文本框
        :return: 竖直翻转之后的图片和文本框
        """
        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 0)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 1] = h - flip_text_polys[:, :, 1]
        return flip_im, flip_text_polys

    def test(self, im: np.ndarray, text_polys: np.ndarray):
        print('随机尺度缩放')
        t_im, t_text_polys = self.random_scale(im, text_polys, [0.5, 1, 2, 3])
        print(t_im.shape, t_text_polys.dtype)
        show_pic(t_im, t_text_polys, 'random_scale')

        print('随机旋转')
        t_im, t_text_polys = self.random_rotate_img_bbox(im, text_polys, 10)
        print(t_im.shape, t_text_polys.dtype)
        show_pic(t_im, t_text_polys, 'random_rotate_img_bbox')

        print('随机裁剪')
        t_im, t_text_polys = self.random_crop_img_bboxes(im, text_polys)
        print(t_im.shape, t_text_polys.dtype)
        show_pic(t_im, t_text_polys, 'random_crop_img_bboxes')

        print('水平翻转')
        t_im, t_text_polys = self.horizontal_flip(im, text_polys)
        print(t_im.shape, t_text_polys.dtype)
        show_pic(t_im, t_text_polys, 'horizontal_flip')

        print('竖直翻转')
        t_im, t_text_polys = self.vertical_flip(im, text_polys)
        print(t_im.shape, t_text_polys.dtype)
        show_pic(t_im, t_text_polys, 'vertical_flip')
        show_pic(im, text_polys, 'vertical_flip_ori')

        print('加噪声')
        t_im = self.add_noise(im)
        print(t_im.shape)
        show_pic(t_im, text_polys, 'add_noise')
        show_pic(im, text_polys, 'add_noise_ori')
