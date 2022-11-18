#!/usr/bin/env python3
import sys
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

from path_from_image.msg import TransformationMatrices

class ImageWarper():

    def __init__(self):
        # transformation matrix for 
        # (un)wraping images to top-view projection
        self.transformMatrix = None
        self.inverseMatrix = None
                 
        # Publishers and subscribers
        # Get topic names from ROS params
        
        self.camera_sub = rospy.Subscriber(
            rospy.get_param('~image_topic'),
            CompressedImage,
            self.camera_callback,
            queue_size=1)
        
        self.img_pub = rospy.Publisher(
            rospy.get_param('~warp_image'),
            CompressedImage,
            queue_size=1)
        
        self.matrix_sub = rospy.Subscriber(
            rospy.get_param('~matrix_topic'),
            TransformationMatrices,
            self.matrix_callback,
            queue_size=1)
        
        rospy.spin()

    def matrix_callback(self, msg):
        """ get transformation matrix from ROS message
        
        Args:
            msg (TransformationMatrices): ROS message with transformation matrix
        """        
        self.transformMatrix = np.array(msg.warp_matrix).reshape(3,3)
        self.inverseMatrix = np.array(msg.inverse_matrix).reshape(3,3)
        # rospy.loginfo('--------------I have transform matrix:')
        # rospy.loginfo('--------------transform matrix: {}'.format(self.transformMatrix))
        # rospy.loginfo('--------------inverse matrix: {}'.format(self.inverseMatrix))

    def camera_callback(self, msg):
        """ get picture and publish its top-view perspective

        Args:
            msg (Image): ros image message
        """        
        if self.transformMatrix is not None:
            # rospy.loginfo('--------------I have already transform matrix and can transform image:')
            try:
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                warp_img = self.warp(cv_image)
                # img type is 8UC4 not compatible with bgr8
                #### Create CompressedIamge ####
                msg = CompressedImage()
                msg.header.stamp = rospy.Time.now()
                msg.format = "jpeg"
                msg.data = np.array(cv2.imencode('.jpg', warp_img)[1]).tobytes()
                self.img_pub.publish(msg)
                    
            except CvBridgeError as e:
                rospy.loginfo(e)
        else:
            rospy.loginfo('--------------There is no transform matrix:')

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
            rospy.loginfo("before warp call set_transform_matrix()")
            return birds_image
        else:
            birds_image = cv2.warpPerspective(np.copy(image), matrix, (w, h))            
        return birds_image

def main(args):
    rospy.init_node('image_warper', anonymous=True, log_level=rospy.INFO)
    node = ImageWarper()

    try:
        print("running image_warper node")
    except KeyboardInterrupt:
        print("Shutting down ROS image_warper node")

if __name__ == '__main__':
    main(sys.argv)