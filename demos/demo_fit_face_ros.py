#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torchvision import transforms

from pixielib.pixie import PIXIE
from pixielib.visualizer import Visualizer
from pixielib.utils.config import cfg as pixie_cfg
from pixielib.utils import util
from skimage.transform import estimate_transform, warp, resize, rescale
from pixielib.datasets import detectors

import numpy as np

class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
        self.image_pub = rospy.Publisher("/output_image_topic", Image, queue_size=10)

        # PIXIE 및 기타 관련 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pixie_cfg.model.use_tex = False
        pixie_cfg.model.iscrop = True
        # self.face_detector = detectors.11;rgb:3030/0a0a/2424FAN()
        self.face_detector = detectors.MP()

        self.pixie = PIXIE(config=pixie_cfg, device=self.device)
        self.pixie_on = None
        self.visualizer = Visualizer(render_size=224, config=pixie_cfg, device=self.device, rasterizer_type="standard")

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

        try:
            processed_image = self.process_image(cv_image)
        except AttributeError as e:
            # print(e)
            return
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(processed_image, "rgb8"))
        except CvBridgeError as e:
            print(e)

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def process_ros_image(self, image):
        # OpenCV 이미지를 numpy 배열로 변환

        h, w, _ = image.shape
        bbox = self.face_detector.run(image)
        image = image[:,:,[2,1,0]]
        if len(bbox) < 4:
            print('no face detected! run original image')
            left = 0; right = h-1; top=0; bottom=w-1
        else:
            left = bbox[0]; right=bbox[2]
            top = bbox[1]; bottom=bbox[3]
        old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')


        size = int(old_size * 2.0)  # self.scale 대신 1.1을 임시로 사용
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])

        crop_size = 224  # 예시로 224를 사용
        DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        dst_image = warp(image, tform.inverse, output_shape=(crop_size, crop_size))
        dst_image = dst_image.transpose(2, 0, 1)

        # 여기에서는 고해상도 이미지 변환을 생략하고, 원본 이미지와 변환된 이미지만 반환합니다.
        return torch.tensor(dst_image).float().unsqueeze(0), tform

    def process_image(self, image):
        # PIXIE 모델 처리
        # OpenCV 이미지를 Tensor로 변환
        origin_image = image.copy()

        start =time.time()
        image_tensor, tform = self.process_ros_image(image)

        image_tensor = image_tensor.to(self.device)
        # Tensor를 사전 형태로 변환
        data = {'head': {'image': image_tensor}}

        # # PIXIE 모델을 사용한 얼굴 3D 재구성
        param_dict = self.pixie.encode(data, keep_local=False, threthold=True)
        codedict = param_dict['head']
        opdict = self.pixie.decode(codedict, param_type='head')
        # opdict['albedo'] = self.visualizer.tex_flame2smplx(opdict['albedo'])

        # 결과 시각화
        visdict = self.visualizer.render_results(opdict, image_tensor, overlay=True)
        # 렌더링 결과를 OpenCV 이미지로 변환
        rendered_image = util.tensor2image(visdict["shape_images"][0])
        h, w, _ = image.shape

        # 렌더링된 이미지를 원본 이미지의 크기에 맞게 변환
        rendered_image_warped = warp(rendered_image, tform, output_shape=(h, w))

        # 렌더링된 이미지를 원본 이미지에 적용
        mask = (rendered_image_warped.sum(axis=-1) != 0)  # 렌더링된 이미지의 마스크 생성
        image[mask] = (rendered_image_warped[mask] * 255).astype(np.uint8)
        image = cv2.addWeighted(origin_image, 0.5, image, 0.5, 2.2)
        end = time.time()
        elapsed = end - start
        # print(elapsed)

        return image


def main():
    rospy.init_node('image_processor', anonymous=True)
    ip = ImageProcessor()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
