import cv2

import rospy
import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry

img_idx = 1


def callback(image_msg, odom_msg):
    global img_idx

    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
    velocity = odom_msg.twist.twist.linear.x

    cv2.imwrite(f'image_{img_idx:05d}_{velocity:.05f}.png', cv_image)
    rospy.loginfo_throttle(5, f'Saved image {img_idx}')
    img_idx += 1


if __name__ == '__main__':
    rospy.init_node('save_images_velocities')
    image_sub = message_filters.Subscriber('/zed2/zed_node/left/image_rect_color', Image)
    odom_sub = message_filters.Subscriber('/odom', Odometry)

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, odom_sub], 10, 0.25)
    ts.registerCallback(callback)

    bridge = CvBridge()

    rospy.spin()
