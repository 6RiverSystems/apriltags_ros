#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <apriltags_ros/nvapriltags/nvAprilTags.h>
#include <apriltags_ros/apriltag_detector_cuda.cuh>
#include <apriltags_ros/nvapriltags/nvAprilTags.h>

namespace apriltags_ros {

    __global__
    void convertUchar3ToUchar4(uchar3 * input, uchar4 * output, int size)
    {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= size)
            {
                return;
            }
            output[index].x = input[index].x;
            output[index].y = input[index].y;
            output[index].z = input[index].z;
            output[index].w = 255;
    }





    class AprilTagDetectorCuda::AprilTagDetectorCudaImpl{
        public:
        std::vector<AprilTags::TagDetection> imageCb(const sensor_msgs::PointCloud2ConstPtr& depth_msg, const sensor_msgs::ImageConstPtr& rgb_msg_in,
		    const sensor_msgs::CameraInfoConstPtr& rgb_info_msg, const sensor_msgs::CameraInfoConstPtr& depth_cam_info);
        private:
        // nvidia april tag gpu detector
        nvAprilTagsHandle hApriltags;
        nvAprilTagsCameraIntrinsics_t cam;
        bool initialized = false;
        nvAprilTagsImageInput_t device_buffer;
        uchar3 * raw_data_ptr;
        uint number_of_elements;
    };

    AprilTagDetectorCuda::AprilTagDetectorCuda() : impl{new AprilTagDetectorCudaImpl()} {}
    AprilTagDetectorCuda::~AprilTagDetectorCuda() {}

    std::vector<AprilTags::TagDetection> AprilTagDetectorCuda::imageCb(const sensor_msgs::PointCloud2ConstPtr& depth_msg, const sensor_msgs::ImageConstPtr& rgb_msg_in,
		    const sensor_msgs::CameraInfoConstPtr& rgb_info_msg, const sensor_msgs::CameraInfoConstPtr& depth_cam_info) {
                return impl->imageCb(depth_msg, rgb_msg_in, rgb_info_msg, depth_cam_info);
            }


    std::vector<AprilTags::TagDetection> AprilTagDetectorCuda::AprilTagDetectorCudaImpl::imageCb(const sensor_msgs::PointCloud2ConstPtr& depth_msg, const sensor_msgs::ImageConstPtr& rgb_msg_in,
		    const sensor_msgs::CameraInfoConstPtr& rgb_info_msg, const sensor_msgs::CameraInfoConstPtr& depth_cam_info) {
        std::vector<AprilTags::TagDetection> detections(10);
        if (!initialized) {
            cam.cx = rgb_info_msg.get()->K[2];
            cam.cy = rgb_info_msg.get()->K[5];
            cam.fx = rgb_info_msg.get()->K[0];
            cam.fy = rgb_info_msg.get()->K[4];
            device_buffer.pitch = 4*rgb_msg_in.get()->width;
            device_buffer.width = rgb_msg_in.get()->width;
            device_buffer.height = rgb_msg_in.get()->height;
            number_of_elements = device_buffer.width * device_buffer.height;
            auto error = cudaMalloc(&device_buffer.dev_ptr, sizeof(uchar4) * number_of_elements);
            if (error != 0) {
                ROS_INFO("GPU memory allocation failed");
                detections.resize(0);
                return detections; 
            }
            error = cudaMalloc(&raw_data_ptr, sizeof(uchar3) * number_of_elements);
            if (error != 0) {
                ROS_INFO("GPU memory allocation failed");
                detections.resize(0);
                return detections; 
            }
            int ret = nvCreateAprilTagsDetector(&hApriltags, rgb_msg_in.get()->width, rgb_msg_in.get()->height, NVAT_TAG36H11, &cam, 0.1);
            if (ret == 0) {
                ROS_INFO("GPU AprilTag Detector initialized");
                initialized = true;
            } else {
                ROS_INFO("GPU Error initializing AprilTag Detector");
                detections.resize(0);
                return detections;
            }
        }
        std::vector<nvAprilTagsID_t> gpu_detections(10);
        std::vector<uchar> test_buffer (number_of_elements * sizeof(uchar4));
        cudaStream_t stream;
        uint32_t count = 0;
        ROS_INFO_STREAM("GPU AprilTag Detector: raw data array size is " << rgb_msg_in.get()->data.size() << " elements");
        ROS_INFO_STREAM("GPU AprilTag Detector: raw data array encoding is " << rgb_msg_in.get()->encoding);

        //cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        ROS_INFO_STREAM("GPU AprilTag Detector: data array size is " << sizeof(uchar4) * number_of_elements << " bytes");
        //cudaMemcpyAsync(static_cast<void *>(device_buffer.dev_ptr), static_cast<const void *>(rgb_msg_in.get()->data.data()), device_buffer.pitch * device_buffer.width * device_buffer.height, cudaMemcpyHostToDevice, stream);
        auto ret = cudaMemcpy(static_cast<void *>(raw_data_ptr), static_cast<const void *>(rgb_msg_in.get()->data.data()), number_of_elements * sizeof(uchar3), cudaMemcpyHostToDevice);
        ROS_INFO_STREAM(__LINE__ << " GPU AprilTag Detector: ret is " << ret);


        dim3 threadsPerBlock(512, 1);
        dim3 numBlocks(static_cast<int>(std::ceil(static_cast<float>(number_of_elements) / threadsPerBlock.x)), 1, 1);
        apriltags_ros::convertUchar3ToUchar4<<<numBlocks, threadsPerBlock>>>(raw_data_ptr, device_buffer.dev_ptr, number_of_elements);
        cudaDeviceSynchronize();
        ret = cudaGetLastError();
        ROS_INFO_STREAM(__LINE__ << " GPU AprilTag Detector: ret is " << ret);
        ROS_INFO_STREAM(__LINE__ << " GPU AprilTag Detector: ret means " << cudaGetErrorString(ret));

        ret = cudaMemcpy(static_cast<void *>(test_buffer.data()), static_cast<const void *>(device_buffer.dev_ptr), sizeof(uchar4) * device_buffer.width * device_buffer.height, cudaMemcpyDeviceToHost);
        ROS_INFO_STREAM(__LINE__ << " GPU AprilTag Detector: ret is " << ret);

        ROS_INFO_STREAM(__LINE__ << " GPU AprilTag Detector: [" << rgb_msg_in.get()->data[0] << ", " << rgb_msg_in.get()->data[1] << ", " << rgb_msg_in.get()->data[2] << "] [" << test_buffer[0] << ", " << test_buffer[1] << ", " << test_buffer[2] << ", " << test_buffer[3] << "]");
        ROS_INFO_STREAM(__LINE__ << " GPU AprilTag Detector: [" << rgb_msg_in.get()->data[3*1024] << ", " << rgb_msg_in.get()->data[3*1024+1] << ", " << rgb_msg_in.get()->data[3*1024+2] << "] [" << test_buffer[4*1024] << ", " << test_buffer[4*1024+1] << ", " << test_buffer[4*1024+2] << ", " << test_buffer[4*1024+3] << "]");
        ROS_INFO_STREAM(__LINE__ << " GPU AprilTag Detector: [" << rgb_msg_in.get()->data[3*(number_of_elements-1)] << ", " << rgb_msg_in.get()->data[3*(number_of_elements-1)+1] << ", " << rgb_msg_in.get()->data[3*(number_of_elements-1)+2] << "] [" << test_buffer[4*(number_of_elements-1)] << ", " << test_buffer[4*(number_of_elements-1)+1] << ", " << test_buffer[4*(number_of_elements-1)+2] << ", " << test_buffer[4*(number_of_elements-1)+3] << "]");


        auto ret1 = nvAprilTagsDetect(hApriltags, &device_buffer, gpu_detections.data(), &count, 10, 0);
        ROS_INFO_STREAM(__LINE__ << " GPU AprilTag Detector: ret1 is " << ret1);
        //cudaStreamSynchronize(stream);
        //cudaStreamDestroy(stream);
        detections.resize(count);
        ROS_INFO_STREAM("GPU AprilTag Detector: " << count << " tags detected");
        for (int tag = 0; tag != count; ++tag) {
            ROS_INFO_STREAM("GPU AprilTag Detector: Tag: " << gpu_detections[tag].id << " detected");
        }
        return detections;
    }

}