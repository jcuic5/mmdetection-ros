#!/usr/bin/env python
"""
 @Author: Jianfeng Cui 
 @Date: 2021-05-22 11:36:30 
 @Last Modified by:   Jianfeng Cui 
 @Last Modified time: 2021-05-22 11:36:30 
"""

# Check Pytorch installation
from logging import debug
from mmcv import image
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import os
import sys
import cv2
import numpy as np

# ROS related imports
import rospy
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image

# NOTE: 
# CvBridge meet problems since we are using python3 env
# We can do the data transformation manually
# from cv_bridge import CvBridge, CvBridgeError

from vision_msgs.msg import Detection2D, \
                            Detection2DArray, \
                            ObjectHypothesisWithPose
from mmdetection_ros.srv import *

from mmdet.models import build_detector

import threading

# Choose to use a config and initialize the detector
CONFIG_NAME = '5_objects.py'
CONFIG_PATH = os.path.join(os.path.dirname(sys.path[0]),'scripts', CONFIG_NAME)

# Setup a checkpoint file to load
MODEL_NAME =  'epoch_12.pth'
MODEL_PATH = os.path.join(os.path.dirname(sys.path[0]),'scripts', MODEL_NAME)

class Detector:

    def __init__(self, model):
        self.image_pub = rospy.Publisher("~debug_image", Image, queue_size=1)
        self.object_pub = rospy.Publisher("~objects", Detection2DArray, queue_size=1)
        # self.bridge = CvBridge()
        self.model = model

        self._last_msg = None
        self._msg_lock = threading.Lock()
        
        self._publish_rate = rospy.get_param('~publish_rate', 1)
        self._is_service = rospy.get_param('~is_service', False)
        self._visualization = rospy.get_param('~visualization', True)

    def generate_obj(self, result, id, msg):
        obj = Detection2D()
        obj.header = msg.header
        obj.source_img = msg
        result = result[0]
        obj.bbox.center.x = (result[0] + result[2]) / 2
        obj.bbox.center.y = (result[1] + result[3]) / 2
        obj.bbox.size_x = result[2] - result[0]
        obj.bbox.size_y = result[3] - result[1]

        obj_hypothesis = ObjectHypothesisWithPose()
        obj_hypothesis.id = str(id)
        obj_hypothesis.score = result[4]
        obj.results.append(obj_hypothesis)

        return obj
        

    def run(self):

        if not self._is_service:
            rospy.loginfo('RUNNING MMDETECTOR AS PUBLISHER NODE')
            image_sub = rospy.Subscriber("~image", Image, self._image_callback, queue_size=1)
        else:
            rospy.loginfo('RUNNING MMDETECTOR AS SERVICE')
            rospy.loginfo('SETTING UP SRV')
            srv = rospy.Service('~image', mmdetSrv, self.service_handler)

        rate = rospy.Rate(self._publish_rate)
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                objArray = Detection2DArray()
                # try:
                #     cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                # except CvBridgeError as e:
                #     print(e)
                # NOTE: This is a way using numpy to convert manually
                im = np.frombuffer(msg.data, dtype = np.uint8).reshape(msg.height, msg.width, -1)
                # image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                image_np = np.asarray(im)

                # Use the detector to do inference
                # NOTE: inference_detector() is able to receive both str and ndarray
                results = inference_detector(self.model, image_np)

                objArray.detections = []
                objArray.header = msg.header
                object_count = 1

                for i in range(len(results)):
                    if results[i].shape != (0, 5):
                        object_count += 1
                        objArray.detections.append(self.generate_obj(results[i], i, msg))

                if not self._is_service:
                    self.object_pub.publish(objArray)
                else:
                    rospy.loginfo('RESPONSING SERVICE')
                    return mmdetSrvResponse(objArray)

                # Visualize results
                if self._visualization:
                    # NOTE: Hack the provided visualization function by mmdetection
                    # Let's plot the result
                    # show_result_pyplot(self.model, image_np, results, score_thr=0.3)
                    # if hasattr(self.model, 'module'):
                    #     m = self.model.module
                    debug_image = self.model.show_result(
                                    image_np,
                                    results,
                                    score_thr=0.3,
                                    show=False,
                                    wait_time=0,
                                    win_name='result',
                                    bbox_color=(72, 101, 241),
                                    text_color=(72, 101, 241))
                    # img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                    # image_out = Image()
                    # try:
                        # image_out = self.bridge.cv2_to_imgmsg(img,"bgr8")
                    # except CvBridgeError as e:
                    #     print(e)
                    # image_out.header = msg.header
                    image_out = msg
                    # NOTE: Copy other fields from msg, modify the data field manually
                    # (check the source code of cvbridge)
                    image_out.data = debug_image.tostring()

                    self.image_pub.publish(image_out)

            rate.sleep()

    def _image_callback(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()

    def service_handler(self, request):
        return self._image_callback(request.image)

def main(args):
    rospy.init_node('mmdetector')
    model = init_detector(CONFIG_PATH, MODEL_PATH, device='cuda:0')
    obj = Detector(model)
    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print("ShutDown")
    obj.run()
    # cv2.destroyAllWindows()

if __name__=='__main__':
    main(sys.argv)