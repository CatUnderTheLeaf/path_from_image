import numpy as np
import cv2
from sympy import Point3D, Line3D, Plane

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
# !!! very important import !!!
# without it Buffer.transform() doesn't work
import tf2_geometry_msgs
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import LookupException
import tf_transformations

from image_geometry import PinholeCameraModel

from custom_msgs.msg import TransformationMatrices

class TransMatrixGetter(Node):

    def __init__(self):
        super().__init__('trans_matrix_getter')        

        # TF stuff
        self._tf_buffer = Buffer()
        self.tf_listener = TransformListener(self._tf_buffer, self)
        # Call on_timer function every second
        self._output_timer = self.create_timer(1.0, self.on_timer)
        self._tf_future = None
        self._when = None

        # get frame names from ROS params
        self.declare_parameter('_camera_frame', 'camera_link_optical')
        self.declare_parameter('_base_frame', 'chassis')
        self._camera_frame = self.get_parameter('_camera_frame').get_parameter_value().string_value
        self._base_frame = self.get_parameter('_base_frame').get_parameter_value().string_value
        
        # Camera stuff
        self.cameraInfoSet = False
        self.camera_model = PinholeCameraModel()
        # transformation matrix for 
        # (un)wraping images to top-view projection
        self.transformMatrix = None
        self.inverseMatrix = None
        # scale factors, image height/width per meters
        self.x_scale = None
        self.y_scale = None
        
        # get these 2 parameters from ROS params
        self.declare_parameter('distance_ahead', 10.0)
        self.declare_parameter('lane_width', 10.0)
        self.distance_ahead = self.get_parameter('distance_ahead').get_parameter_value().double_value
        self.lane_width = self.get_parameter('lane_width').get_parameter_value().double_value
        
        # Publishers and subscribers
        # Get topic names from ROS params
        self.declare_parameter('transf_matrix', '/path/transf_matrix')
        self.declare_parameter('camera_info', '/vehicle/front_camera/camera_info')

        self.matrix_pub = self.create_publisher(
            TransformationMatrices,
            self.get_parameter('transf_matrix').get_parameter_value().string_value,
            10)
        self.matrix_pub
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info').get_parameter_value().string_value,
            self.info_callback,
            1)
        self.camera_info_sub
        
    def on_tf_ready(self, future):
        """ set transformation matrix when tf-transform is ready

        Args:
            future (Future): the outcome of a transform-task in the future
        """        
        self._tf_future = None
        if (future.result() and self.cameraInfoSet):
            try:
                self.set_transform_matrix()
            except LookupException:
                self.get_logger().info('transform no longer available')
            # else:
            #     self.get_logger().info('Got transform')

    def on_timer(self):
        """ call and wait for tf-transformation to be ready
        """        
        # self.get_logger().info('on_timer function called')
        if self._tf_future:
            # self.get_logger().info('Still waiting for transform')
            return
        self._when = rclpy.time.Time()
        self._tf_future = self._tf_buffer.wait_for_transform_async(
            self._camera_frame, self._base_frame, self._when)
        self._tf_future.add_done_callback(self.on_tf_ready)
        # self.get_logger().info('Waiting for transform from {} to {}'.format(
        #     self._base_frame, self._camera_frame))

    def info_callback(self, msg):   
        """ get camera info and load it to the camera model

        Args:
            msg (CameraInfo): ros camera info message
        """             
        if not self.cameraInfoSet:
            # self.get_logger().info('I heard first cameraInfo------------------')
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
        q = tf_transformations.quaternion_from_euler(0., 0., 0)
        p.pose.orientation.x = q[0]
        p.pose.orientation.y = q[1]
        p.pose.orientation.z = q[2]
        p.pose.orientation.w = q[3]
        p.header.frame_id = source_frame
        p.header.stamp = self.get_clock().now().to_msg()
        
        # apply transformation to a pose between source_frame and dest_frame
        newPoint = self._tf_buffer.transform(p, dest_frame)
        pose = newPoint.pose.position        
        return pose.x, pose.y, pose.z

    def set_transform_matrix(self):
        """find src and dst points for CV2 perspective transformation
            and set transformMatrix
        """               
        if not (self.camera_model):
            self.get_logger().info("camera_model is not set")
            return
        else:
            h = self.camera_model.height
            w = self.camera_model.width
            # left bottom corner        
            lbc_ray = self.camera_model.projectPixelTo3dRay((0, h))
            # right bottom corner        
            rbc_ray = self.camera_model.projectPixelTo3dRay((w, h))
            # camera center and bottom points in the robot/world frame
            zero = self.transformPoint((0.,0.,0.), fromCamera=True)
            lbc_point = self.transformPoint(lbc_ray, fromCamera=True)
            rbc_point = self.transformPoint(rbc_ray, fromCamera=True)            
            point3, point4, x_scale = self.getUpperPoints(zero, lbc_point, rbc_point)   
            # set scale factors in pixel/meters
            self.x_scale = w/self.lane_width
            self.y_scale = h/self.distance_ahead
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
            if (src and dst):
                self.transformMatrix = self.get_transform_matrix(src, dst)
                self.inverseMatrix = self.get_transform_matrix(dst, src)
                matrices = TransformationMatrices()
                matrices.warp_matrix = self.transformMatrix.flatten()
                matrices.inverse_matrix = self.inverseMatrix.flatten()
                self.matrix_pub.publish(matrices)
            return    
   
    def get_transform_matrix(self, src, dst):
        """get cv2 transform matrix for perspective transformation

        Args:
            src (list): list of 4 points in the source image
            dst (list): list of 4 points in the destination image

        Returns:
            matrix: transformation matrix
        """        
        # get matrix for perspective transformation
        transformMatrix = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
        return transformMatrix

    def getUpperPoints(self, zero, lbc, rbc):
        """ get upper src points in the robot/world frame

        Args:
            zero (tuple): camera center
            lbc (tuple): ray which goes from center and left bottom corner of the image
            rbc (tuple): ray which goes from center and right bottom corner of the image

        Returns:
            tuple: two upper points and x_scale factor
        """        
        # make two lines from camera center
        lbc_line = Line3D(Point3D(zero), Point3D(lbc))
        rbc_line = Line3D(Point3D(zero), Point3D(rbc))
        
        # ground plane with lanes
        xoy = (0.,0., lbc[2])
        xy_plane = Plane(Point3D(xoy), normal_vector=(0, 0, 1))
        
        # bottom points in the robot/world frame
        # are intersection points of lines and ground plane
        point1 = xy_plane.intersection(lbc_line)[0]
        point2 = xy_plane.intersection(rbc_line)[0]

        # distance in meters between bottom points
        # after their projection onto the ground plane
        x_scale = float(point1.distance(point2))

        # get upper points in the robot/world frame
        # sometimes in car models front camera 
        # looks in the negative y-axe direction
        # so our y-translation should has right sign
        # TODO maybe has to be some way to determine 
        # which axis goes along the car model and
        # if it has the same direction as z-axis of the cv2-image
        if lbc[1] > 0:
            sign = 1
        else:
            sign = -1
        point3 = point1.translate(0, sign*self.distance_ahead)
        point4 = point2.translate(0, sign*self.distance_ahead)
        
        return point3, point4, x_scale




def main(args=None):
    rclpy.init(args=args)
    trans_matrix_getter = TransMatrixGetter()

    rclpy.spin(trans_matrix_getter)

    trans_matrix_getter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()