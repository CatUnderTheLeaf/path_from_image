import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


from cv_bridge import CvBridge, CvBridgeError

class LaneAreaDrawer(Node):

    def __init__(self):
        super().__init__('lane_area_drawer')      
        
        # transformation matrix for 
        # (un)wraping images to top-view projection
        self.transformMatrix = None
        # scale factors, image height/width per meters
        self.x_scale = None
        self.y_scale = None
        
        # get these 2 parameters from ROS params
        self.declare_parameter('distance_ahead', 10.0)
        self.declare_parameter('lane_width', 10.0)
        self.distance_ahead = self.get_parameter('distance_ahead').get_parameter_value().double_value
        self.lane_width = self.get_parameter('lane_width').get_parameter_value().double_value
        
        self.bridge = CvBridge()

        # Publishers and subscribers
        # Get topic names from ROS params
        self.declare_parameter('wrap_img', '/wrap_img')
        self.declare_parameter('lane_image', '/lane_image')       

        self.camera_sub = self.create_subscription(
            Image,
            self.get_parameter('wrap_img').get_parameter_value().string_value,
            self.camera_callback,
            10)
        self.camera_sub
        self.img_pub = self.create_publisher(
            Image,
            self.get_parameter('lane_image').get_parameter_value().string_value,
            10)
        self.img_pub

    def camera_callback(self, msg):
        """ get picture and publish its top-view perspective

        Args:
            msg (Image): ros image message
        """        
        if self.transformMatrix is not None:
            self.get_logger().info('--------------I have already transform matrix and can transform image:')
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                lane_img = self.drawLaneArea(cv_image)
                # img type is 8UC4 not compatible with bgr8
                lane_img_msg = self.bridge.cv2_to_imgmsg(lane_img, "bgr8")
                self.img_pub.publish(lane_img_msg)
                    
            except CvBridgeError as e:
                self.get_logger().info(e)
        else:
            self.get_logger().info('--------------There is no transform matrix:')

    def drawLaneArea(self, cv_image):
        """Draw lane area on top of the image
        
        Args:
            cv_image (OpenCV image): image to be drawn on
        
        Returns:
            np.array: image with lane area drawn on it
        """        
        # get the tresholded image
        binary = self.treshold_binary(cv_image)
        # get the perspective transform matrix
        # self.transformMatrix = self.getPerspectiveTransformMatrix(cv_image)
        # get the scale factors
        # self.x_scale, self.y_scale = self.getScaleFactors(cv_image)
        # get the lane area
        # lane_area = self.getLaneArea(binary)
        # draw the lane area
        # lane_area_img = self.drawLaneAreaOnImage(cv_image, lane_area)
        lane_area_img = binary
        return lane_area_img

    def treshold_binary(self, image, s_thresh=(50, 255), sx_thresh=(20, 100)):
        """Gradient and color tresholding of an image 

        Args:
            image (OpenCV image): image to be tresholded
            s_thresh (tuple, optional): _description_. Defaults to (50, 255).
            sx_thresh (tuple, optional): _description_. Defaults to (20, 100).

        Returns:
            np.array: binary image
        """        
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
         
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx)) if (np.max(abs_sobelx)) else np.uint8(255*abs_sobelx)
        
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        # Threshold color channel
        # in simulation h and s channels are zeros
        if (np.max(s_channel) == 0):
            s_channel = l_channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        binary = np.dstack((combined_binary, combined_binary, combined_binary)) * 255
        self.treshold_image = binary

        return binary




def main(args=None):
    rclpy.init(args=args)
    lane_area_drawer = LaneAreaDrawer()

    rclpy.spin(lane_area_drawer)

    lane_area_drawer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()