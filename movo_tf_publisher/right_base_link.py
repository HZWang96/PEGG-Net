#!/usr/bin/env python

import rospy
import tf
import geometry_msgs.msg

if __name__ == '__main__':
    rospy.init_node('right_ee_state_publisher')
    listener = tf.TransformListener()
    right_to_right_base_info = rospy.Publisher('right_ee_to_right_base', geometry_msgs.msg.Transform,queue_size=1)
    right_to_base_info = rospy.Publisher('right_ee_state', geometry_msgs.msg.Transform,queue_size=1)

    rate = rospy.Rate(150)
    while not rospy.is_shutdown():
        try:
            (trans_r,rot_r) = listener.lookupTransform('/right_base_link','/right_ee_link', rospy.Time(0))
            (trans_base,rot_base) = listener.lookupTransform('/base_link','/right_ee_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        print("sending tf from right ee link to right_base_link and base_link")
        right_trans = geometry_msgs.msg.Transform()
        right_trans.translation.x= trans_r[0]
        right_trans.translation.y= trans_r[1]
        right_trans.translation.z= trans_r[2]
        right_trans.rotation.x=rot_r[0]
        right_trans.rotation.y=rot_r[1]
        right_trans.rotation.z=rot_r[2]
        right_trans.rotation.w=rot_r[3]
        right_to_right_base_info.publish(right_trans)

        right_to_base_trans = geometry_msgs.msg.Transform()
        right_to_base_trans.translation.x= trans_base[0]
        right_to_base_trans.translation.y= trans_base[1]
        right_to_base_trans.translation.z= trans_base[2]
        right_to_base_trans.rotation.x=rot_base[0]
        right_to_base_trans.rotation.y=rot_base[1]
        right_to_base_trans.rotation.z=rot_base[2]
        right_to_base_trans.rotation.w=rot_base[3]
        right_to_base_info.publish(right_to_base_trans)
        rate.sleep()
