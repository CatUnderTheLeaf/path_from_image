import numpy as np
import cv2
import rclpy
import threading
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
# !!! very important import !!!
# without it Buffer.transform doesn't work
import tf2_geometry_msgs
from cv_bridge import CvBridge, CvBridgeError
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from tf2_ros import LookupException
import tf_transformations
from sympy import Point3D, Line3D, Plane

from image_geometry import PinholeCameraModel

class ImageWarper(Node):

    def __init__(self):
        super().__init__('image_warper')        

        # TF stuff
        self._tf_buffer = Buffer()
        self.tf_listener = TransformListener(self._tf_buffer, self)
        # Call on_timer function every second
        self._output_timer = self.create_timer(1.0, self.on_timer)
        self._lock = threading.Lock()
        self._tf_future = None
        # TODO get from params
        self._camera_frame = 'camera_link_optical'
        self._base_frame = 'chassis'
        self._when = None

        # Camera stuff
        self.cameraInfoSet = False
        self.camera_model = PinholeCameraModel()
        # transformation matrix for 
        # (un)wraping images to top-view projection
        self.transformMatrix = None
        # scale factors, meters per image height/width
        self.x_scale = None
        # TODO 
        # get scale parameter from params server
        # y_scale = rospy.get_param('~y_scale')
        self.y_scale = 10.0


        # mock of CameraInfo
        info = CameraInfo()
        info.header.frame_id = 'front_camera_sensor'
        info.height = 800
        info.width = 800
        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [476.7030836014194, 0.0, 400.5, 0.0, 476.7030836014194, 400.5, 0.0, 0.0, 1.0]
        info.p = [476.7030836014194, 0.0, 400.5, -33.36921585209936, 0.0, 476.7030836014194, 400.5, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.binning_x = 0
        info.binning_y = 0
        info.roi.x_offset = 0
        info.roi.y_offset = 0
        info.roi.height = 0
        info.roi.width = 0
        info.roi.do_rectify = False
        self.camera_model.fromCameraInfo(info)
        self.cameraInfoSet = True



        self.bridge = CvBridge()

        # Publishers and subscribers
        # self.camera_sub = self.create_subscription(
        #     Image,
        #     '/vehicle/front_camera/image_raw',
        #     self.camera_callback,
        #     10)
        # self.camera_sub
        # self.img_pub = self.create_publisher(
        #     Image,
        #     '/wrap_img',
        #     10)
        # self.img_pub
        # self.camera_info_sub = self.create_subscription(
        #     CameraInfo,
        #     '/vehicle/front_camera/camera_info',
        #     self.info_callback,
        #     1)
        # self.camera_info_sub
        
    def on_tf_ready(self, future):
        self._tf_future = None
        if (future.result() and self.cameraInfoSet):
            try:
                self.set_transform_matrix()
            except LookupException:
                self.get_logger().info('transform no longer available')
            else:
                # b = trans.transform.translation
                self.get_logger().info('Got transform')

    def on_timer(self):
        self.get_logger().info('on_timer function called')
        if self._tf_future:
            self.get_logger().info('Still waiting for transform')
            return
        if self.transformMatrix is not None:
            self.get_logger().info('transform matrix is set, no need to wait for transforms:')
        else:
            self._when = rclpy.time.Time()
            self._tf_future = self._tf_buffer.wait_for_transform_async(
                self._camera_frame, self._base_frame, self._when)
            self._tf_future.add_done_callback(self.on_tf_ready)
            self.get_logger().info('Waiting for transform from {} to {}'.format(
                self._base_frame, self._camera_frame))
   
    def camera_callback(self, msg):
        if self.transformMatrix is not None:
            self.get_logger().info('--------------I have already transform matrix and can transform image:')
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                # import os
                # import cv2
                # test_dir_path = '/home/catundertheleaf/robocar/robocar_ws/src/path_from_image/test'
                # self.get_logger().info(test_dir_path)
                # cv2.imwrite(os.path.join(test_dir_path, 'cv_image.jpg'), cv_image)
                warp_img = self.warp(cv_image)
                # img type is 8UC4 not compatible with bgr8
                top_view_msg = self.bridge.cv2_to_imgmsg(warp_img, "bgr8")
                self.img_pub.publish(top_view_msg)
                    
            except CvBridgeError as e:
                self.get_logger().info(e)
        else:
            self.get_logger().info('--------------There is no transform matrix:')

    def info_callback(self, msg):        
        if not self.cameraInfoSet:
            self.get_logger().info('I heard first cameraInfo------------------')
            self.camera_model.fromCameraInfo(msg)
            self.cameraInfoSet = True

    def warp(self, image):
        """wrap image into top-view perspective

        Args:
            image (cv2_img): image to transform

        Returns:
            cv2_img: wraped image
        """        
        h, w = image.shape[0], image.shape[1]
        birds_image = image
        if (self.transformMatrix is None):
            self.get_logger().info("before warp call set_transform_matrix()")
            return birds_image
        else:
            birds_image = cv2.warpPerspective(np.copy(image), self.transformMatrix, (w, h))            
        return birds_image

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
            # transform points 3 and 4 to camera_optical_link frame
            luc_point = self.transformPoint(point3, fromCamera=False)
            ruc_point = self.transformPoint(point4, fromCamera=False)
            # pixel coordinates of these points:
            # left upper corner 
            luc = self.camera_model.project3dToPixel(luc_point)
            # right upper corner        
            ruc = self.camera_model.project3dToPixel(ruc_point)
            # form src and dst points
            # TODO make rotation to the pose, because x axes are not in the same directions
            self.get_logger().info("zero point in base frame")
            self.get_logger().info(f'zero {float(zero[0])}, {float(zero[1])}, {float(zero[2])}')
            self.get_logger().info("lbc and rbc in base frame")
            self.get_logger().info(f'lbc {float(lbc_point[0])}, {float(lbc_point[1])}, {float(lbc_point[2])}')
            self.get_logger().info(f'rbc {float(rbc_point[0])}, {float(rbc_point[1])}, {float(rbc_point[2])}')
            self.get_logger().info("luc and ruc in base frame")
            self.get_logger().info(f'luc {float(point3[0])}, {float(point3[1])}, {float(point3[2])}')
            self.get_logger().info(f'ruc {float(point4[0])}, {float(point4[1])}, {float(point4[2])}')
            # self.get_logger().info(f'approx luc {w/3}, {h/2}')
            # self.get_logger().info(f'approx ruc {2*w/3}, {h/2}')
            src = [[luc[0], luc[1]],[ruc[0], ruc[1]],[w, h],[0, h]]
            # src = [[w/3, h/2],[2*w/3, h/2],[w, h],[0, h]]
            dst = [[0, 0],[w, 0],[w, h],[0, h]]
            # dst = [[0, 0],[w, 0],[2*w/3, h],[w/3, h]]
            if (src and dst):
                self.transformMatrix = self.get_transform_matrix(src, dst)
                # self.x_scale = x_scale/w
                # self.y_scale = y_scale/h
                # TODO
                # mock transform image
                test_path_in = '/home/catundertheleaf/robocar/robocar_ws/src/path_from_image/path_from_image/cv_image.jpg'
                img = cv2.imread(test_path_in)
                out_img = self.warp(img)
                test_path_out = '/home/catundertheleaf/robocar/robocar_ws/src/path_from_image/path_from_image/wraped_image.jpg'
                cv2.imwrite(test_path_out, out_img)
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

        # translate factor in meters
        # depends on h*w of the image
        # and scale factor
        x_scale = float(point1.distance(point2))

        # get upper points in the robot/world frame
        # sometimes in car models front camera 
        # looks in the negative y-axe direction
        # so our y-translation should has right sign
        if lbc[1] > 0:
            sign = 1
        else:
            sign = -1
        point3 = point1.translate(0, sign*self.y_scale)
        point4 = point2.translate(0, sign*self.y_scale)
        
        return point3, point4, x_scale




def main(args=None):
    rclpy.init(args=args)
    # TODO get params
    # camera_link_optical = rospy.get_param('~camera_opt_frame')
    # base_bottom_link = rospy.get_param('~base_frame')
    image_warper = ImageWarper()

    rclpy.spin(image_warper)

    image_warper.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()