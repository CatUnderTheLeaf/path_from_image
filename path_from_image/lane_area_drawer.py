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
        left_fit, right_fit, img_with_windows = self.fit_polynomial(binary)
        draw_img = self.draw_filled_polygon(cv_image, left_fit, right_fit)
        # draw_img = draw_polylines(out_img, left_fit, right_fit)
        # draw_img = draw_filled_polygon(out_img, left_fit, right_fit)
        # get the perspective transform matrix
        # self.transformMatrix = self.getPerspectiveTransformMatrix(cv_image)
        # get the scale factors
        # self.x_scale, self.y_scale = self.getScaleFactors(cv_image)
        # get the lane area
        # lane_area = self.getLaneArea(binary)
        # draw the lane area
        # lane_area_img = self.drawLaneAreaOnImage(cv_image, lane_area)
        lane_area_img = draw_img
        return lane_area_img

    # draw a green polygon between two lanes
    def draw_filled_polygon(self, img, left_fit, right_fit):
        draw_img = np.copy(img)
        ploty, left_fitx, right_fitx = self.get_xy(draw_img.shape[0], left_fit, right_fit)
        
        all_x = np.concatenate([left_fitx, np.flip(right_fitx, 0)])
        all_y = np.concatenate([ploty, np.flip(ploty, 0)])
        all_points = [(np.asarray([all_x, all_y]).T).astype(np.int32)]
        cv2.fillPoly(draw_img, all_points, (0,255,0))
        
        return draw_img

    def get_xy(self, shape_0, left_fit, right_fit):
        ploty = np.linspace(0, shape_0-1, shape_0 )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty
            
        return ploty, left_fitx, right_fitx

    def fit_polynomial(self, treshold_warped):
        """return polynomial for both lanes

        Args:
            treshold_warped (CVimage): tresholded binary image

        Returns:
            tuple: polynomials for right and left lane
        """        
        h, w = treshold_warped.shape[0], treshold_warped.shape[1]
        # ym_per_pix, xm_per_pix - meters per pixel in x or y dimension
        ym_per_pix=10.0/h
        xm_per_pix=10.0/w
        left_fit = [0,0,0]
        right_fit = [0,0,0]
            
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(treshold_warped, True)
        
        # check if the image is not empty
        if (leftx.size > 0 and lefty.size > 0 and rightx.size > 0 and righty.size > 0):
            left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
            right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)      
        
        return left_fit, right_fit, out_img

    def find_lane_pixels(self, image, draw=False):
        """find lane pixels in image with sliding windows

        Args:
            image (CVimage): tresholded binary image
            draw (bool, optional): draw rectangles on the image or not. Defaults to False.

        Returns:
            tuple: right and left lane pixels
        """    
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 20
        # Set the width of the windows +/- margin
        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix = 200

        treshold_warped = image[:,:,0]
        out_img = np.copy(image)
        # Take a histogram of the bottom half of the image
        histogram = np.sum(treshold_warped[treshold_warped.shape[0]//2:,:], axis=0)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(histogram, distance=300)
        # if the image is empty
        if (np.amax(histogram) < 10):
            leftx = lefty = rightx = righty = np.array([])
        else:        
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int64(histogram.shape[0]//2)
            leftx_base = np.argmax(histogram[:midpoint-margin])
            rightx_base = np.argmax(histogram[midpoint+margin:]) + midpoint  
            leftx_base = peaks[0]
            rightx_base = peaks[-1]

            # Set height of windows - based on nwindows above and image shape
            window_height = np.int64(treshold_warped.shape[0]//nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = treshold_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current positions to be updated later for each window in nwindows
            leftx_current = leftx_base
            rightx_current = rightx_base

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = treshold_warped.shape[0] - (window+1)*window_height
                win_y_high = treshold_warped.shape[0] - window*window_height
                
                # Find the four below boundaries of the window 
                win_xleft_low = leftx_current - margin 
                win_xleft_high = leftx_current + margin                     
                # draw window rectangles
                if draw:
                    cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                    (win_xleft_high,win_y_high),(0,255,0), 2) 
                # Identify the nonzero pixels in x and y within the window 
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                # If you found > minpix pixels, recenter next window on their mean position 
                # or stop windows
                if len(good_left_inds) > minpix:
                    leftx_current=np.int64(np.mean(nonzerox[good_left_inds]))

                # Find the four below boundaries of the window 
                win_xright_low = rightx_current - margin 
                win_xright_high = rightx_current + margin                      
                # draw window rectangles
                if draw:
                    cv2.rectangle(out_img,(win_xright_low,win_y_low),
                    (win_xright_high,win_y_high),(0,255,0), 2)  

                # Identify the nonzero pixels in x and y within the window                     
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position 
                # or stop windows
                if (good_right_inds.shape[0] > minpix):
                    rightx_current=np.int64(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices (previously was a list of lists of pixels)
            try:
                left_lane_inds = np.concatenate(left_lane_inds)
                right_lane_inds = np.concatenate(right_lane_inds)
            except ValueError:
                # Avoids an error if the above is not implemented fully
                pass

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            out_img[lefty, leftx] = [0, 255, 0]
            out_img[righty, rightx] = [255, 0, 0]

        return leftx, lefty, rightx, righty, out_img 

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