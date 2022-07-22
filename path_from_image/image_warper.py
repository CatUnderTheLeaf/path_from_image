import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from custom_msgs.msg import TransformationMatrices

class ImageWarper(Node):

    def __init__(self):
        super().__init__('image_warper')      
        
        # transformation matrix for 
        # (un)wraping images to top-view projection
        self.transformMatrix = None
        self.inverseMatrix = None
                 
        self.bridge = CvBridge()

        # Publishers and subscribers
        # Get topic names from ROS params
        self.declare_parameter('image_raw', '/vehicle/front_camera/image_raw')
        self.declare_parameter('warp_image', '/path/warp_image')
        self.declare_parameter('transf_matrix', '/path/transf_matrix')

        self.camera_sub = self.create_subscription(
            Image,
            self.get_parameter('image_raw').get_parameter_value().string_value,
            self.camera_callback,
            10)
        self.camera_sub
        
        self.img_pub = self.create_publisher(
            Image,
            self.get_parameter('warp_image').get_parameter_value().string_value,
            10)
        self.img_pub
        
        self.matrix_sub = self.create_subscription(
            TransformationMatrices,
            self.get_parameter('transf_matrix').get_parameter_value().string_value,
            self.matrix_callback,
            10)
        self.matrix_sub

    def matrix_callback(self, msg):
        """ get transformation matrix from ROS message
        
        Args:
            msg (TransformationMatrices): ROS message with transformation matrix
        """        
        self.transformMatrix = msg.warp_matrix.reshape(3,3)
        self.inverseMatrix = msg.inverse_matrix.reshape(3,3)
        # self.get_logger().info('--------------I have transform matrix:')
        # self.get_logger().info('--------------transform matrix: {}'.format(self.transformMatrix))
        # self.get_logger().info('--------------inverse matrix: {}'.format(self.inverseMatrix))

    def camera_callback(self, msg):
        """ get picture and publish its top-view perspective

        Args:
            msg (Image): ros image message
        """        
        if self.transformMatrix is not None:
            # self.get_logger().info('--------------I have already transform matrix and can transform image:')
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                warp_img = self.warp(cv_image)
                # img type is 8UC4 not compatible with bgr8
                warp_img_msg = self.bridge.cv2_to_imgmsg(warp_img, "bgr8")
                self.img_pub.publish(warp_img_msg)
                    
            except CvBridgeError as e:
                self.get_logger().info(e)
        else:
            self.get_logger().info('--------------There is no transform matrix:')

    def warp(self, image, top_view=True):      
        """wrap image into top-view perspective

        Args:
            image (cv2_img): image to transform
            top_view (bool, optional): warp into top-view perspective or vice versa. Defaults to True.

        Returns:
            cv2_img: wraped image
        """        
        h, w = image.shape[0], image.shape[1]
        birds_image = image
        matrix = self.transformMatrix
        if not top_view:
            matrix = self.inverseMatrix
        if (matrix is None):
            self.get_logger().info("before warp call set_transform_matrix()")
            return birds_image
        else:
            birds_image = cv2.warpPerspective(np.copy(image), matrix, (w, h))            
        return birds_image

def main(args=None):
    rclpy.init(args=args)
    image_warper = ImageWarper()

    rclpy.spin(image_warper)

    image_warper.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()