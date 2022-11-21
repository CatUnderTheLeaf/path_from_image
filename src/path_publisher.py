#!/usr/bin/env python3
import sys
import numpy as np
import cv2
from sympy import Point3D, Line3D, Plane

import rospy
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped, Polygon
from nav_msgs.msg import Path
# !!! very important import !!!
# without it Buffer.transform() doesn't work
import tf2_geometry_msgs
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf.transformations import quaternion_from_euler

from image_geometry import PinholeCameraModel

class PathPublisher():

    def __init__(self):
        
        # TF stuff
        self._tf_buffer = Buffer()
        self.tf_listener = TransformListener(self._tf_buffer)
        self.canTransform = None

        # get frame names from ROS params
        self._camera_frame = rospy.get_param('~_camera_frame')
        self._base_frame = rospy.get_param('~_base_frame')
        
        # Camera stuff
        self.cameraInfoSet = False
        self.camera_model = PinholeCameraModel()
           
        # Publishers and subscribers
        # Get topic names from ROS params
        
        self.path_pub = rospy.Publisher(
            rospy.get_param('~path_waypoints'),
            Path,
            queue_size=1)
        
        self.camera_info_sub = rospy.Subscriber(
            rospy.get_param('~info_topic'),
            CameraInfo,
            self.info_callback,
            queue_size=1)
        
        self.img_waypoints_sub = rospy.Subscriber(
            rospy.get_param('~img_waypoints'),
            Polygon,
            self.img_waypoints_callback,
            queue_size=1)

        rospy.spin()    

    def info_callback(self, msg):   
        """ get camera info and load it to the camera model

        Args:
            msg (CameraInfo): ros camera info message
        """             
        if not self.cameraInfoSet:
            # rospy.loginfo('I heard first cameraInfo------------------')
            self.camera_model.fromCameraInfo(msg)
            self.cameraInfoSet = True
        if not self.canTransform:
            try:
                if self._tf_buffer.can_transform(self._camera_frame, self._base_frame, rospy.Time()):
                    self.canTransform = True
            except (LookupException, ConnectivityException, ExtrapolationException):
                rospy.loginfo('I cannnot transform------------------')

    def img_waypoints_callback(self, msg):
        """ get waypoints from image and publish them as path

        Args:
            msg (Polygon): ros polygon message with waypoints
        """        
        rospy.loginfo('img callback------------------')
        if self.canTransform and self.cameraInfoSet:
            rospy.loginfo('I can transform------------------')
            waypoints = []
            zero = self.transformPoint((0.,0.,0.))

            # transform waypoints from image frame to base frame
            for point in msg.points:
                ray = self.camera_model.projectPixelTo3dRay((point.x, point.y))
                waypoints.append(self.getWaypoint(zero, self.transformPoint(ray)))
            
            # Create and publish path
            path = Path()
            path.header.frame_id = self._base_frame
            path.header.stamp = rospy.Time.now()
            path.poses = waypoints

            self.path_pub.publish(path)
        
            
    def getWaypoint(self, zero, transformed_point):
        """Get a waypoint in the base frame

        Args:
            zero (tuple): camera center in the robot/world frame
            transformed_point (tuple): image waypoint in the robot/world frame

        Returns:
            PoseStamped: waypoint in the base frame
        """ 
        # TODO: !!!!!!! very slow make smth else    
        line = Line3D(Point3D(zero), Point3D(transformed_point))
        
        # ground plane with lanes
        xoy = (0.,0., 0.)
        xy_plane = Plane(Point3D(xoy), normal_vector=(0, 0, 1))
        
        # the point in the robot/world frame
        # is an intersection point of a line and ground plane
        new_point = xy_plane.intersection(line)[0]

        # Form the PoseStamped
        waypoint = PoseStamped()
        waypoint.pose.position.x = float(new_point[0])
        waypoint.pose.position.y = float(new_point[1])
        waypoint.pose.position.z = float(new_point[2])
        q = quaternion_from_euler(0., 0., 0)
        waypoint.pose.orientation.x = q[0]
        waypoint.pose.orientation.y = q[1]
        waypoint.pose.orientation.z = q[2]
        waypoint.pose.orientation.w = q[3]
        waypoint.header.frame_id = self._base_frame
        waypoint.header.stamp = rospy.Time.now()
        # rospy.loginfo(f'new point {new_point[0]}, {new_point[1]}, {new_point[2]}')
        return waypoint

    def transformPoint(self, point):
        """transform point from one to another frame

        Args:
            point (tuple): a point in 3d space

        Returns:
            tuple: point in the new frame
        """        
        # only PointStamped, PoseStamped, PoseWithCovarianceStamped, Vector3Stamped, PointCloud2
        # can be transformed between frames
        p = PoseStamped()
        p.pose.position.x = float(point[0])
        p.pose.position.y = float(point[1])
        p.pose.position.z = float(point[2])
        q = quaternion_from_euler(0., 0., 0)
        p.pose.orientation.x = q[0]
        p.pose.orientation.y = q[1]
        p.pose.orientation.z = q[2]
        p.pose.orientation.w = q[3]
        p.header.frame_id = self._camera_frame
        p.header.stamp = rospy.Time.now()
        
        # apply transformation to a pose between source_frame and dest_frame
        newPoint = self._tf_buffer.transform(p, self._base_frame)
        pose = newPoint.pose.position    
        # rospy.loginfo(f'old point {point[0]}, {point[1]}, {point[2]}')   
        # rospy.loginfo(f'new point {pose.x}, {pose.y}, {pose.z}') 
        return pose.x, pose.y, pose.z

def main(args):
    rospy.init_node('path_publisher', anonymous=True, log_level=rospy.INFO)
    node = PathPublisher()

    try:
        print("running path_publisher node")
    except KeyboardInterrupt:
        print("Shutting down ROS path_publisher node")

if __name__ == '__main__':
    main(sys.argv)
