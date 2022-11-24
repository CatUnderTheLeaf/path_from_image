#!/usr/bin/env python3
import sys
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Point32, Polygon

from path_from_image.msg import TransformationMatrices
from path_from_image import lane_finder

class LaneAreaDrawer():

    def __init__(self):        
        # transformation matrix for 
        # (un)wraping images to top-view projection
        self.transformMatrix = None
        self.inverseMatrix = None
        self.cameraInfo = None
        self.undistortMaps = (None, None)
       
        # Publishers and subscribers
        # Get topic names from ROS params

        self.camera_info_sub = rospy.Subscriber(
            rospy.get_param('~info_topic'),
            CameraInfo,
            self.info_callback,
            queue_size=1)

        self.camera_sub = rospy.Subscriber(
            rospy.get_param('~image_topic'),
            CompressedImage,
            self.camera_callback,
            queue_size=1)
        
        self.img_pub = rospy.Publisher(
            rospy.get_param('~lane_image'),
            CompressedImage,
            queue_size=1)
        
        self.matrix_sub = rospy.Subscriber(
            rospy.get_param('~matrix_topic'),
            TransformationMatrices,
            self.matrix_callback,
            queue_size=1)
        
        self.waypoint_pub = rospy.Publisher(
            rospy.get_param('~img_waypoints'),
            Polygon,
            queue_size=1)
        
        rospy.spin()

    def info_callback(self, msg):   
        """ get camera info and load it to the camera model

        Args:
            msg (CameraInfo): ros camera info message
        """             
        if not self.cameraInfo:
            rospy.logdebug('load cameraInfo------------------')
            self.cameraInfo = msg 
            m = np.array(self.cameraInfo.K).reshape(3,3)
            d = np.array(self.cameraInfo.D)
            size = (self.cameraInfo.width, self.cameraInfo.height)
            r = np.array(self.cameraInfo.R).reshape(3,3)
            self.undistortMaps = cv2.initUndistortRectifyMap(m, d, r, m, size, cv2.CV_32FC1)  

    def matrix_callback(self, msg):
        """ get transformation matrix from ROS message
        
        Args:
            msg (TransformationMatrices): ROS message with transformation matrix
        """        
        rospy.logdebug('--------------warp_matrix: {}'.format(msg.warp_matrix))
        self.transformMatrix = np.array(msg.warp_matrix).reshape(3,3)
        self.inverseMatrix = np.array(msg.inverse_matrix).reshape(3,3)
        rospy.logdebug('--------------I have transform matrix:')
        rospy.logdebug('--------------transform matrix: {}'.format(self.transformMatrix))
        rospy.logdebug('--------------inverse matrix: {}'.format(self.inverseMatrix))

    def camera_callback(self, msg):
        """ get picture and publish it with added lane area
            also publish points of the middle lane line in the image

        Args:
            msg (Image): ros image message
        """        
        if (self.transformMatrix is not None) and (self.cameraInfo is not None):
            rospy.logdebug('--------------I have already transform matrix and can transform image:')
            
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # rectify image
            map1, map2 = self.undistortMaps
            undist = cv2.remap(cv_image, map1, map2, cv2.INTER_LINEAR)

            lane_img, img_waypoints = lane_finder.drawLaneArea(undist,self.transformMatrix, self.inverseMatrix)
            # make image message and publish it
            # img type is 8UC4 not compatible with bgr8
            #### Create CompressedIamge ####
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tobytes()
            msg.data = np.array(cv2.imencode('.jpg', lane_img)[1]).tobytes()
            self.img_pub.publish(msg)
            # make polygon message and publish it
            img_waypoints_msg = self.make_polygon_msg(img_waypoints)
            self.waypoint_pub.publish(img_waypoints_msg)
            rospy.logdebug('--------------publish waypoints:')
            
        else:
            rospy.logdebug('--------------There is no transform matrix:')

    def make_polygon_msg(self, points):
        """make polygon message from points
        
        Args:
            points (list): list of points
        
        Returns:
            Polygon: polygon message
        """        
        polygon = Polygon()
        polygon.points = [Point32(x=p[0], y=p[1], z=0.0) for p in points[0]]
        return polygon

def main(args):
    rospy.init_node('lane_area_drawer', anonymous=True, log_level=rospy.INFO)
    node = LaneAreaDrawer()

    try:
        print("running lane_area_drawer node")
    except KeyboardInterrupt:
        print("Shutting down ROS lane_area_drawer node")

if __name__ == '__main__':
    main(sys.argv)
