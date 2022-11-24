#include "ros/ros.h"
#include <image_transport/image_transport.h>
// #include "sensor_msgs/CompressedImage.h"
#include "sensor_msgs/CameraInfo.h"
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

bool cameraInfoSet=false;
cv::Mat map1, map2;
image_transport::Publisher pub;

void imageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
  ROS_INFO("I heard: image");
  if (cameraInfoSet) {
    ROS_INFO("I heard: ");
    cv::Mat image_cv = cv_bridge::toCvCopy(msg)->image;
    cv::Mat m_undistImg;
    cv::remap(image_cv, m_undistImg, map1, map2, cv::INTER_LINEAR );
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", m_undistImg).toImageMsg();
    pub.publish(msg);
  }
}

void infoCallback(const sensor_msgs::CameraInfo& camInfo_msg)
{
  ROS_INFO("I heard: info");
  if (!cameraInfoSet) {
    cv::Mat intrK(3, 3, CV_64FC1);
    cv::Mat dist(5, 1, CV_64FC1);
    intrK.setTo(0);
    intrK.at<double>(0,0) = camInfo_msg.K[0];
    intrK.at<double>(0,1) = camInfo_msg.K[1];
    intrK.at<double>(0,2) = camInfo_msg.K[2];
    intrK.at<double>(1,0) = camInfo_msg.K[3];
    intrK.at<double>(1,1) = camInfo_msg.K[4];
    intrK.at<double>(1,2) = camInfo_msg.K[5];
    intrK.at<double>(2,0) = camInfo_msg.K[6];
    intrK.at<double>(2,1) = camInfo_msg.K[7];
    intrK.at<double>(2,2) = camInfo_msg.K[8];
    for(int i=0; i<5; ++i)
            dist.at<double>(i, 0) = 0;
    
    cv::initUndistortRectifyMap(intrK, dist, cv::Mat(), intrK, cv::Size(camInfo_msg.width,camInfo_msg.height), CV_32FC1, map1, map2);
    cameraInfoSet = true;
    }
  }

int main(int argc, char **argv)
{

  ros::init(argc, argv, "undistort_node");
  ros::NodeHandle nh_params("~");

  std::string camera_info_topic;
  std::string image_topic;  
  std::string undistort_image_topic;
  nh_params.param("image_topic", image_topic, std::string("/raspicam/image/"));
  nh_params.param("info_topic", camera_info_topic, std::string("/raspicam/camera_info"));
  nh_params.param("undistort_image_topic", undistort_image_topic, std::string("/raspicam/undst_image"));


  ros::NodeHandle n;
  image_transport::ImageTransport it(n);

  image_transport::Subscriber image_sub = it.subscribe(image_topic, 1, imageCallback);
  pub = it.advertise(undistort_image_topic, 1);
  ros::Subscriber camera_info_sub = n.subscribe(camera_info_topic, 1, infoCallback);


  ros::spin();

  return 0;
}