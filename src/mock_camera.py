#!/usr/bin/env python
import sys
import cv2
import os

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class MockCamera():

    def __init__(self):        
        # mock of CameraInfo
        self.info = CameraInfo()
        self.info.header.frame_id = 'camera_link_optical'
        self.info.height = 308
        self.info.width = 410
        self.info.distortion_model = 'plumb_bob'
        self.info.D = [-3.02767206e-01, 9.00556422e-02, 1.22879175e-04, 5.87956322e-04, -1.16976357e-02]
        self.info.K = [330.85633761, 0, 342.10606951, 0, 329.88375819, 252.34461849, 0, 0, 1]
        self.info.P = [330.37335205, 0, 341.60664969, 0, 0, 329.24319458, 251.8546135, 0, 0, 0, 1, 0]
        self.info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.info.binning_x = 0
        self.info.binning_y = 0
        self.info.roi.x_offset = 0
        self.info.roi.y_offset = 0
        self.info.roi.height = 0
        self.info.roi.width = 0
        self.info.roi.do_rectify = False

        self.bridge = CvBridge()

        # Publishers
        # Get topic names from ROS params
       
        self.camera_pub = rospy.Publisher(
            rospy.get_param('~image_topic'),
            Image,
            queue_size=10)

        self.camera_info_pub = rospy.Publisher(
            rospy.get_param('~info_topic'),
            CameraInfo,
            queue_size=10)

        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            self.on_timer()
            rate.sleep()

    def on_timer(self):
        try:
            test_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # rospy.loginfo('test_dir_path: {}'.format(os.path.join(test_dir_path, 'resource', 'img1.png')))
            camera_img = cv2.imread(os.path.join(test_dir_path, 'resource', 'img2.png'))
            # make image message and publish it
            # img type is 8UC4 not compatible with bgr8
            camera_img_msg = self.bridge.cv2_to_imgmsg(camera_img, "bgr8")
            self.camera_pub.publish(camera_img_msg)
            self.camera_info_pub.publish(self.info)
                
        except CvBridgeError as e:
            self.get_logger().info(e)

def main(args):
    rospy.init_node('mock_camera_node', anonymous=True, log_level=rospy.INFO)
    node = MockCamera()

    try:
        print("running mock_camera node")
    except KeyboardInterrupt:
        print("Shutting down ROS mock_camera node")

if __name__ == '__main__':
    main(sys.argv)