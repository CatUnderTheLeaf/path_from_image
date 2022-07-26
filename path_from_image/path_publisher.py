from sympy import Point3D, Line3D, Plane

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped, Polygon
from nav_msgs.msg import Path
# !!! very important import !!!
# without it Buffer.transform() doesn't work
import tf2_geometry_msgs
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import LookupException
import tf_transformations

from image_geometry import PinholeCameraModel

class PathPublisher(Node):

    def __init__(self):
        super().__init__('path_publisher')        

        # TF stuff
        self._tf_buffer = Buffer()
        self.tf_listener = TransformListener(self._tf_buffer, self)

        # get frame names from ROS params
        self.declare_parameter('_camera_frame', 'camera_link_optical')
        self.declare_parameter('_base_frame', 'chassis')
        self._camera_frame = self.get_parameter('_camera_frame').get_parameter_value().string_value
        self._base_frame = self.get_parameter('_base_frame').get_parameter_value().string_value
        
        # Camera stuff
        self.cameraInfoSet = False
        self.camera_model = PinholeCameraModel()
            
        # Publishers and subscribers
        # Get topic names from ROS params
        self.declare_parameter('img_waypoints', '/path/img_waypoints')
        self.declare_parameter('path_waypoints', '/path/path_waypoints')
        self.declare_parameter('camera_info', '/vehicle/front_camera/camera_info')

        self.path_pub = self.create_publisher(
            Path,
            self.get_parameter('path_waypoints').get_parameter_value().string_value,
            10)
        self.path_pub
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info').get_parameter_value().string_value,
            self.info_callback,
            1)
        self.camera_info_sub
        self.img_waypoints_sub = self.create_subscription(
            Polygon,
            self.get_parameter('img_waypoints').get_parameter_value().string_value,
            self.img_waypoints_callback,
            10)
        self.img_waypoints_sub

    def info_callback(self, msg):   
        """ get camera info and load it to the camera model

        Args:
            msg (CameraInfo): ros camera info message
        """             
        if not self.cameraInfoSet:
            # self.get_logger().info('I heard first cameraInfo------------------')
            self.camera_model.fromCameraInfo(msg)
            self.cameraInfoSet = True

    def img_waypoints_callback(self, msg):
        """ get waypoints from image and publish them as path

        Args:
            msg (Polygon): ros polygon message with waypoints
        """        
        canTransform = self._tf_buffer.can_transform(self._camera_frame, self._base_frame, rclpy.time.Time())
        
        if canTransform and self.cameraInfoSet:
            self.get_logger().info('I can transform------------------')
            waypoints = []
            zero = self.transformPoint((0.,0.,0.))
            # transform waypoints from image frame to base frame
            for point in msg.points:
                waypoints.append(self.getWaypoint(zero, self.transformPoint((point.x, point.y, point.z))))
            
            # Create and publish path
            path = Path()
            path.header.frame_id = self._base_frame
            path.header.stamp = self.get_clock().now().to_msg()
            path.poses = waypoints

            self.path_pub.publish(path)
        else:
            self.get_logger().info('I cannnot transform------------------')

    def getWaypoint(self, zero, transformed_point):
        """Get a waypoint in the base frame

        Args:
            zero (tuple): camera center in the robot/world frame
            transformed_point (tuple): image waypoint in the robot/world frame

        Returns:
            PoseStamped: waypoint in the base frame
        """        
        line = Line3D(Point3D(zero), Point3D(transformed_point))
        
        # ground plane with lanes
        xoy = (0.,0., transformed_point[2])
        xy_plane = Plane(Point3D(xoy), normal_vector=(0, 0, 1))
        
        # the point in the robot/world frame
        # is an intersection point of a line and ground plane
        new_point = xy_plane.intersection(line)[0]

        # Form the PoseStamped
        waypoint = PoseStamped()
        waypoint.pose.position.x = float(new_point[0])
        waypoint.pose.position.y = float(new_point[1])
        waypoint.pose.position.z = float(new_point[2])
        q = tf_transformations.quaternion_from_euler(0., 0., 0)
        waypoint.pose.orientation.x = q[0]
        waypoint.pose.orientation.y = q[1]
        waypoint.pose.orientation.z = q[2]
        waypoint.pose.orientation.w = q[3]
        waypoint.header.frame_id = self._base_frame
        waypoint.header.stamp = self.get_clock().now().to_msg()

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
        x, y, z = point
        p.pose.position.x = float(x)
        p.pose.position.y = float(y)
        p.pose.position.z = float(z)
        q = tf_transformations.quaternion_from_euler(0., 0., 0)
        p.pose.orientation.x = q[0]
        p.pose.orientation.y = q[1]
        p.pose.orientation.z = q[2]
        p.pose.orientation.w = q[3]
        p.header.frame_id = self._camera_frame
        p.header.stamp = self.get_clock().now().to_msg()
        
        # apply transformation to a pose between source_frame and dest_frame
        newPoint = self._tf_buffer.transform(p, self._base_frame)
        pose = newPoint.pose.position        
        return pose.x, pose.y, pose.z

def main(args=None):
    rclpy.init(args=args)
    path_publisher = PathPublisher()

    rclpy.spin(path_publisher)

    path_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()