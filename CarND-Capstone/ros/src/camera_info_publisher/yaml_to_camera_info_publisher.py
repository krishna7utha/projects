#!/usr/bin/env python

import rospy
import yaml
from sensor_msgs.msg import CameraInfo

def yaml_to_CameraInfo(calib_yaml):

    # Load data from file
    calib_data = yaml.load(calib_yaml)
    # Parse
    camera_info_msg = CameraInfo()
    camera_info_msg.width = calib_data["image_width"]
    camera_info_msg.height = calib_data["image_height"]
    camera_info_msg.K = calib_data["camera_matrix"]["data"]
    camera_info_msg.D = calib_data["distortion_coefficients"]["data"]
    camera_info_msg.R = calib_data["rectification_matrix"]["data"]
    camera_info_msg.P = calib_data["projection_matrix"]["data"]
    camera_info_msg.distortion_model = calib_data["distortion_model"]
    return camera_info_msg

if __name__ == "__main__":

    calib_yaml = rospy.get_param("/grasshopper_calibration_yaml")

    # Parse yaml file
    camera_info_msg = yaml_to_CameraInfo(calib_yaml)

    # Initialize publisher node
    rospy.init_node("camera_info_publisher", anonymous=True)
    publisher = rospy.Publisher("camera_info", CameraInfo, queue_size=10)
    rate = rospy.Rate(10)

    # Run publisher
    while not rospy.is_shutdown():
        publisher.publish(camera_info_msg)
        rate.sleep()
