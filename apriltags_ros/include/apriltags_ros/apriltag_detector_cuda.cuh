#pragma once

#include <ros/ros.h>
#include <std_msgs/Int8.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <message_filters/sync_policies/exact_time.h>
#include <AprilTags/TagDetector.h>
#include <tf/transform_broadcaster.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <chrono>
#include <apriltags_ros/AprilTagDetection.h>
#include <memory>

namespace apriltags_ros {

    class AprilTagDetectorCuda{
        public:
        AprilTagDetectorCuda();
        ~AprilTagDetectorCuda();

        std::vector<AprilTags::TagDetection> imageCb(const sensor_msgs::PointCloud2ConstPtr& depth_msg, const sensor_msgs::ImageConstPtr& rgb_msg_in,
		    const sensor_msgs::CameraInfoConstPtr& rgb_info_msg, const sensor_msgs::CameraInfoConstPtr& depth_cam_info);
        private:
        // nvidia april tag gpu detector
        class AprilTagDetectorCudaImpl;
        std::unique_ptr<AprilTagDetectorCudaImpl> impl;
    };
}