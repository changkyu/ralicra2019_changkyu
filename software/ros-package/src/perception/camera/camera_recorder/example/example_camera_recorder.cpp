#include "camera_recorder/camera_recorder.hpp"

int main(int argc, char** argv)
{
	ros::init(argc,argv,"package_random_push");
    ros::NodeHandle nh;
	CameraRecorder rec(nh, getenv("HOME"), "example");

	rec.SaveCurrentFrame();
}