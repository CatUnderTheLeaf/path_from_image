import rclpy
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
import tf_transformations

from path_from_image.top_viewer import topViewer

class ImageWarper(Node):

    def __init__(self):
        super().__init__('image_warper')
        # TODO get from params
        self.camera_link_optical = 'camera'
        self.base_bottom_link = 'base_link'
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cameraInfoSet = False
        self.topViewer = topViewer()
        self.hasTransformMatrix = False
        self.bridge = CvBridge()
        self.camera_sub = self.create_subscription(
            Image,
            '/vehicle/front_camera/image_raw',
            self.camera_callback,
            10)
        self.camera_sub
        self.img_pub = self.create_publisher(
            Image,
            '/wrap_img',
            10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/vehicle/front_camera/camera_info',
            self.info_callback,
            1)
        self.camera_info_sub
        # Call on_timer function every second
        self.timer = self.create_timer(1.0, self.on_timer)
        
    def on_timer(self):
        if self.cameraInfoSet and (not self.hasTransformMatrix):
            try:
                self.set_transform_matrix(self.camera_link_optical, self.base_bottom_link)
                self.get_logger().info('--------------im getting transform matrix:')
            except TransformException as ex:
                self.get_logger().info('--------------Could not transform')
                return              

    def camera_callback(self, msg):
        if self.hasTransformMatrix:
            self.get_logger().info('--------------I have already transform matrix:')
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                warp_img = self.topViewer(cv_image)
                # img type is 8UC4 not compatible with bgr8
                top_view_msg = self.bridge.cv2_to_imgmsg(warp_img, "bgr8")
                self.img_pub.publish(top_view_msg)
                    
            except CvBridgeError as e:
                print(e)
        else:
            self.get_logger().info('--------------There is no transform matrix:')


    def info_callback(self, msg):        
        if not self.cameraInfoSet:
            self.get_logger().info('I heard first cameraInfo------------------')
            self.topViewer.camera_model.fromCameraInfo(msg)
            self.cameraInfoSet = True
    
    def set_transform_matrix(self, camera_link_optical, base_bottom_link):
        # find src and dst points
        # it is done here because points 
        # should be transformed between frames with tfTransformer
        h = self.topViewer.camera_model.height
        w = self.topViewer.camera_model.width
        # left bottom corner        
        lbc_ray = self.topViewer.camera_model.projectPixelTo3dRay((0, h))
        # right bottom corner        
        rbc_ray = self.topViewer.camera_model.projectPixelTo3dRay((w, h))
        # camera center and bottom points in the base_bottom_link frame
        zero = self.get_point_xyz(self.transformPoint((0.,0.,0.), camera_link_optical, base_bottom_link))
        lbc_point = self.get_point_xyz(self.transformPoint(lbc_ray, camera_link_optical, base_bottom_link))
        rbc_point = self.get_point_xyz(self.transformPoint(rbc_ray, camera_link_optical, base_bottom_link))
        # get scale parameter from params server
        # TODO what to do with x_scale?
        # y_scale = rospy.get_param('~y_scale')
        # point3, point4, x_scale = self.top_viewer.getUpperPoints(zero, lbc_point, rbc_point, y_scale)
        # self.x_translate = point3[0] - y_scale
        point3, point4, x_scale = self.topViewer.getUpperPoints(zero, lbc_point, rbc_point)
        # transform points 3 and 4 to camera_optical_link frame
        luc_point = self.get_point_xyz(self.transformPoint(point3, base_bottom_link, camera_link_optical))
        ruc_point = self.get_point_xyz(self.transformPoint(point4, base_bottom_link, camera_link_optical))
        # pixel coordinates of these points:
        # left upper corner 
        luc = self.topViewer.camera_model.project3dToPixel(luc_point)
        # right upper corner        
        ruc = self.topViewer.camera_model.project3dToPixel(ruc_point)
        # form src and dst points
        src = [[luc[0], luc[1]],[ruc[0], ruc[1]],[w, h],[0, h]]
        dst = [[0, 0],[w, 0],[w, h],[0, h]]
        if (src and dst):
            self.topViewer.get_transform_matrix(src, dst)
            # self.top_viewer.setScale((x_scale/w, y_scale/h))
            self.hasTransformMatrix = True

    def transformPoint(self, point, source_frame, dest_frame, translate=0.):
        # make geometry_msgs/PoseStamped Message
        p = PoseStamped()
        p.pose.position.x = float(point[0] + translate)
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
        new_point = self.tf_buffer.transform(p, dest_frame)
        return new_point

    def get_point_xyz(self, point):
        p = point.pose.position        
        return p.x, p.y, p.z

def main(args=None):
    rclpy.init(args=args)
    # TODO get params
    # camera_link_optical = rospy.get_param('~camera_opt_frame')
    # base_bottom_link = rospy.get_param('~base_frame')
    image_warper = ImageWarper()

    rclpy.spin(image_warper)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_warper.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()