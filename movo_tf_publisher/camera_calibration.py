#!/usr/bin/env python  
import rospy
import tf
import pickle

if __name__ == '__main__':
    rospy.init_node('right_calibration')
    br = tf.TransformBroadcaster()
    listener = tf.TransformListener()
    a_file = open("camera_calibration_record.pkl", "rb")
    tf_dict = pickle.load(a_file)
    ww = tf_dict["ww"]
    xx = tf_dict["xx"]
    yy = tf_dict["yy"]
    zz = tf_dict["zz"]
    x = tf_dict["x"]
    y = tf_dict["y"]
    z = tf_dict["z"]
    pass
r = rospy.Rate(100)
while not rospy.is_shutdown():
    print("sending tf from realsense to right_ee_link")
    br.sendTransform((x, y, z),
                    (xx, yy, zz, ww),
                    rospy.Time.now(),
                    "realsense_depth_optical_frame",
                    "right_ee_link")
    r.sleep()
