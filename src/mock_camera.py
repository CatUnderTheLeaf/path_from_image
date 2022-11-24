#!/usr/bin/env python
import sys
import cv2
import os
import numpy as np

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo

class MockCamera():

    def __init__(self):        
        # mock of CameraInfo
        self.info = CameraInfo()
        self.info.header.frame_id = 'camera_link_optical'
        self.info.height = 308
        self.info.width = 410
        self.info.distortion_model = 'plumb_bob'
        self.info.D = [-3.28296006e-01, 1.19443754e-01, -1.85799276e-04, 8.39998127e-04, -2.06502314e-02]
        self.info.K = [202.35126146, 0, 206.80143037, 0, 201.08542928, 153.32497517, 0, 0, 1]
        self.info.P = [201.85769653, 0, 206.29701676, 0, 0, 200.43255615, 152.82716951, 0, 0, 0, 1, 0]
        self.info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.info.binning_x = 0
        self.info.binning_y = 0
        self.info.roi.x_offset = 0
        self.info.roi.y_offset = 0
        self.info.roi.height = 0
        self.info.roi.width = 0
        self.info.roi.do_rectify = False


        # Publishers
        # Get topic names from ROS params
       
        self.camera_pub = rospy.Publisher(
            rospy.get_param('~image_topic'),
            CompressedImage,
            queue_size=1)

        self.camera_info_pub = rospy.Publisher(
            rospy.get_param('~info_topic'),
            CameraInfo,
            queue_size=1)

        rate = rospy.Rate(30) # 10hz
        while not rospy.is_shutdown():
            self.on_timer()
            rate.sleep()

    def on_timer(self):
        
        test_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # rospy.loginfo('test_dir_path: {}'.format(os.path.join(test_dir_path, 'resource', 'img1.png')))
        camera_img = cv2.imread(os.path.join(test_dir_path, 'resource', 'frame0002.jpg'))
        # make image message and publish it
        # img type is 8UC4 not compatible with bgr8
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', camera_img)[1]).tobytes()
        self.camera_pub.publish(msg)
        self.camera_info_pub.publish(self.info)
        

def main(args):
    rospy.init_node('mock_camera_node', anonymous=True, log_level=rospy.INFO)
    node = MockCamera()

    try:
        print("running mock_camera node")
    except KeyboardInterrupt:
        print("Shutting down ROS mock_camera node")

if __name__ == '__main__':
    main(sys.argv)