#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

image_transport::Publisher image_pub;
ros::Publisher info_pub;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    ros::Time ros_time = ros::Time::now(); 
    sensor_msgs::Image newmsg = sensor_msgs::Image();
    newmsg.header = msg->header;
    newmsg.header.stamp = ros_time; 
    newmsg.height = msg->height;
    newmsg.width = msg->width;
    newmsg.encoding = msg->encoding;
    newmsg.is_bigendian = msg->is_bigendian;
    newmsg.step = msg->step;
    newmsg.data = msg->data;

    sensor_msgs::CameraInfo cfg;
    std::vector<double> list;
    int i;

    cfg.header.frame_id = "camera";
    cfg.header.stamp = ros_time; 
    double temp;
    ros::param::get("/image_height", temp);
    cfg.height = (uint)temp;
    ros::param::get("/image_width", temp);
    cfg.width = (uint)temp;
    ros::param::get("/distortion_model", cfg.distortion_model);
    ros::param::get("/distortion_coefficients/data", list);
    cfg.D.clear();
    for (i=0;i<5;i++) {
        cfg.D.push_back((double)list[i]);
    }
    ros::param::get("/camera_matrix/data", list);
    for (i=0;i<9;i++) {
        cfg.K[i] = list[i];
    }
    ros::param::get("/rectification_matrix/data", list);
    for (i=0;i<9;i++) {
        cfg.R[i] = list[i];
    }
    ros::param::get("/projection_matrix/data", list);
    for (i=0;i<12;i++) {
        cfg.P[i] = list[i];
    }

    image_pub.publish(newmsg);
    info_pub.publish(cfg);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "republish_image_info");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  ros::NodeHandle nh_params("~");

  std::string camera_info_topic;
  std::string image_topic;  
  std::string repub_image_topic;
  nh_params.param("image_topic", image_topic, std::string("/raspicam/image/"));
  nh_params.param("camera_info_topic", camera_info_topic, std::string("/camera/camera_info"));
  nh_params.param("repub_image_topic", repub_image_topic, std::string("/camera/image"));
  image_pub = it.advertise(repub_image_topic, 1);
  info_pub = nh.advertise<sensor_msgs::CameraInfo>(camera_info_topic, 1);
  
  image_transport::Subscriber sub = it.subscribe(image_topic, 1, imageCallback);

  ros::spin();
}