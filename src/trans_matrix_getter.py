#!/usr/bin/env python3
import sys
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
# !!! very important import !!!
# do not forget to import tf2_geometry_msgs
# without it Buffer.transform() 
# from one frame to another doesn't work
import tf2_geometry_msgs
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf.transformations import quaternion_from_euler

from image_geometry import PinholeCameraModel

from path_from_image.msg import TransformationMatrices
from path_from_image import geometry_helper

class TransMatrixGetter():

    def __init__(self):
        
        # TF stuff
        self._tf_buffer = Buffer()
        self.tf_listener = TransformListener(self._tf_buffer)
        
        # get frame names from ROS params
        self._camera_frame = rospy.get_param('~_camera_frame')
        self._base_frame = rospy.get_param('~_base_frame')
        
        # Camera stuff
        self.cameraInfoSet = False
        self.camera_model = PinholeCameraModel()
        # transformation matrix for 
        # (un)wraping images to top-view projection
        self.matrixSet = False
        self.transformMatrix = None
        self.inverseMatrix = None
        # scale factors, image height/width per meters
        self.x_scale = None
        self.y_scale = None
        
        # get these 2 parameters from ROS params
        self.distance_ahead = rospy.get_param('~distance_ahead')
        self.lane_width = rospy.get_param('~lane_width')

        # Publishers and subscribers
        # Get topic names from ROS params        
        self.matrix_pub = rospy.Publisher(
            rospy.get_param('~matrix_topic'),
            TransformationMatrices,
            queue_size=1)
        
        self.camera_info_sub = rospy.Subscriber(
            rospy.get_param('~camera_info'),
            CameraInfo,
            self.info_callback,
            queue_size=1)

        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            if (self.cameraInfoSet):
                if not self.matrixSet:
                    try:
                        if self._tf_buffer.can_transform(self._camera_frame, self._base_frame, rospy.Time()):
                            self.set_transform_matrix()
                    except (LookupException, ConnectivityException, ExtrapolationException):
                        rate.sleep()
                        continue
                else:
                    matrices = TransformationMatrices()
                    matrices.warp_matrix = self.transformMatrix.flatten()
                    matrices.inverse_matrix = self.inverseMatrix.flatten()
                    self.matrix_pub.publish(matrices)
                    rospy.logdebug('matrix is ready------------------')  
                    rospy.logdebug('{}'.format(matrices.warp_matrix)) 
                    rospy.logdebug('{}'.format(matrices.inverse_matrix))  
                rate.sleep()

    def info_callback(self, msg):   
        """ get camera info and load it to the camera model

        Args:
            msg (CameraInfo): ros camera info message
        """             
        if not self.cameraInfoSet:
            rospy.logdebug('I heard first cameraInfo------------------')
            self.camera_model.fromCameraInfo(msg)
            self.cameraInfoSet = True
      
    def transformPoint(self, point, fromCamera=True):
        """transform point from one to another frame

        Args:
            point (tuple): a point in 3d space
            fromCamera (bool): if True, transform point from the camera frame and vice versa

        Returns:
            tuple: point in the new frame
        """        
        
        if fromCamera:
            source_frame = self._camera_frame
            dest_frame = self._base_frame
        else:
            source_frame = self._base_frame
            dest_frame = self._camera_frame
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
        p.header.frame_id = source_frame
        p.header.stamp = rospy.Time.now()
        
        # apply transformation to a pose between source_frame and dest_frame
        newPoint = self._tf_buffer.transform(p, dest_frame)
        pose = newPoint.pose.position        
        return pose.x, pose.y, pose.z

    def set_transform_matrix(self):
        """find src and dst points for CV2 perspective transformation
            and set transformMatrix
        """         
        rospy.logdebug('setting transform matrix------------------')      
        if not (self.camera_model):
            rospy.logdebug("camera_model is not set")
            return
        else:
            h = self.camera_model.height
            w = self.camera_model.width
            rospy.logdebug("image width={} and height={}".format(w, h))
            # left bottom corner        
            lbc_ray = self.camera_model.projectPixelTo3dRay((0, h))
            # right bottom corner        
            rbc_ray = self.camera_model.projectPixelTo3dRay((w, h))
            rospy.logdebug("lbc_ray in camera frame (x, y, z) ({})".format(lbc_ray))         
            rospy.logdebug("rbc_ray point in camera frame (x, y, z) ({})".format(rbc_ray))         
            
            # camera center and bottom points in the robot/world frame
            zero = self.transformPoint((0.,0.,0.), fromCamera=True)
            lbc_point = self.transformPoint(lbc_ray, fromCamera=True)
            rbc_point = self.transformPoint(rbc_ray, fromCamera=True)   
            rospy.logdebug("zero point in robot frame (x, y, z) ({})".format(zero))         
            rospy.logdebug("lbc_point point in robot frame (x, y, z) ({})".format(lbc_point))         
            rospy.logdebug("rbc_point point in robot frame (x, y, z) ({})".format(rbc_point))         
            point3, point4, x_scale = geometry_helper.getUpperPoints(zero, lbc_point, rbc_point, self.distance_ahead)
            rospy.logdebug("point3 point in robot frame (x, y, z) ({})".format(point3))         
            rospy.logdebug("point4 point in robot frame (x, y, z) ({})".format(point4))         
               
            # set scale factors in pixel/meters
            self.x_scale = w/self.lane_width
            self.y_scale = h/self.distance_ahead
            rospy.logdebug("image x_scale={} and y_scale={}".format(self.x_scale, self.y_scale))
            
            # transform points 3 and 4 to camera_optical_link frame
            luc_point = self.transformPoint(point3, fromCamera=False)
            ruc_point = self.transformPoint(point4, fromCamera=False)
            # pixel coordinates of these points:
            # left upper corner 
            luc = self.camera_model.project3dToPixel(luc_point)
            # right upper corner        
            ruc = self.camera_model.project3dToPixel(ruc_point)
            # form src and dst points
            # left upper corner, right upper corner, right bottom corner, left bottom corner
            src = [[luc[0], luc[1]],[ruc[0], ruc[1]],[w, h],[0, h]]
            # new x coordinates are scaled with lane_width
            l_w = w/2*(1-x_scale/self.lane_width)
            r_w = w/2*(1+x_scale/self.lane_width)
            dst = [[l_w, 0],[r_w, 0],[r_w, h],[l_w, h]]
            rospy.logdebug("src points = [[{}, {}], [{}, {}], [{}, {}], [0, {}]".format(luc[0], luc[1], ruc[0], ruc[1], w, h, h))
            rospy.logdebug("src points = [[{}, 0], [{}, 0], [{}, {}], [0, {}]".format(l_w, r_w, r_w, h, l_w, h))
            
            if (src and dst):
                self.transformMatrix = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
                self.inverseMatrix = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))             
                self.matrixSet = True
            return    


def main(args):
    rospy.init_node('trans_matrix_getter', anonymous=True, log_level=rospy.INFO)
    node = TransMatrixGetter()

    try:
        print("running trans_matrix_getter node")
    except KeyboardInterrupt:
        print("Shutting down ROS trans_matrix_getter node")

if __name__ == '__main__':
    main(sys.argv)