import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

'''
For cropping body:
1. using bbox from objection detectors
2. calculate bbox from body joints regressor

object detectors:
    know body object from label number
    https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    label for peopel: 1
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
'''
# TODO: add hand detector

#-- detetion
class FasterRCNN(object):
    ''' detect body
    '''
    def __init__(self, device='cuda:0'):  
        '''
        https://pytorch.org/docs/stable/torchvision/models.html#faster-r-cnn
        '''
        import torchvision
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.device = device
    @torch.no_grad()
    def run(self, input):
        '''
        input: 
            The input to the model is expected to be a list of tensors, 
            each of shape [C, H, W], one for each image, and should be in 0-1 range. 
            Different images can have different sizes.
        return: 
            detected box, [x1, y1, x2, y2]
        '''
        prediction = self.model(input.to(self.device))[0]
        inds = (prediction['labels']==1)*(prediction['scores']>0.5)
        if len(inds) < 1:
            return None
        else:
            bbox = prediction['boxes'][inds][0].cpu().numpy()
            return bbox
    
    @torch.no_grad()
    def run_multi(self, input):
        '''
        input: 
            The input to the model is expected to be a list of tensors, 
            each of shape [C, H, W], one for each image, and should be in 0-1 range. 
            Different images can have different sizes.
        return: 
            detected box, [x1, y1, x2, y2]
        '''
        prediction = self.model(input.to(self.device))[0]
        inds = (prediction['labels']==1)*(prediction['scores']>0.9)
        if len(inds) < 1:
            return None
        else:
            bbox = prediction['boxes'][inds].cpu().numpy()
            return bbox

# TODO
class Yolov4(object):
    def __init__(self, device='cuda:0'):
        pass

    @torch.no_grad()
    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        pass

#-- person keypoints detection
# tested, not working well
class KeypointRCNN(object):
    ''' Constructs a Keypoint R-CNN model with a ResNet-50-FPN backbone.
    Ref: https://pytorch.org/docs/stable/torchvision/models.html#keypoint-r-cnn
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    '''
    def __init__(self, device='cuda:0'):  
        import torchvision
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def run(self, input):
        '''
        input: 
            The input to the model is expected to be a list of tensors, 
            each of shape [C, H, W], one for each image, and should be in 0-1 range. 
            Different images can have different sizes.
        return: 
            boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x between 0 and W and values of y between 0 and H
            labels (Int64Tensor[N]): the class label for each ground-truth box
            keypoints (FloatTensor[N, K, 3]): the K keypoints location for each of the N instances, in the format [x, y, visibility], where visibility=0 means that the keypoint is not visible.
        '''
        prediction = self.model(input.to(self.device))[0]
        # 
        kpt = prediction['keypoints'][0].cpu().numpy()
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
        bbox = [left, top, right, bottom]
        return bbox, kpt

#-- face landmarks (68)
class FAN(object):
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda', face_detector='blazeface')

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0]
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            return bbox

class MP(object):
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='/home/s-kim/tmp/PIXIE/pixielib/datasets/detector.tflite',
                                          delegate=python.BaseOptions.Delegate.CPU)
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        out = self.detector.detect(mp_image)
        bbox = out.detections[0].bounding_box
        # start_point = bbox.origin_x, bbox.origin_y
        # end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

        if out is None:
            return [0]
        else:
            left = bbox.origin_x
            right = left + bbox.height
            top = bbox.origin_y
            bottom = top + bbox.width
            return [left, top, right, bottom]
