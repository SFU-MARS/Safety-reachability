## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

def EuclidDistance(wpt1, wpt2):
    return np.sqrt( (wpt1[0] - wpt2[0]) ** 2 + (wpt1[1] - wpt2[1]) ** 2 )

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# World-frame x-y waypoints
# TODO: This should be streamed directly from another thread that computes the SVM boundary
#wpts_list = [[-2., 3.], [-5., 2.]]

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        #depth_image = np.asanyarray(depth_frame.get_data())
        rgb_image = np.asanyarray(color_frame.get_data())

        """
        List of things to do:
            - Convert the wpts_list into image frame
            - Plot the wpts onto the image
            - Detect clicking events for the plotted waypoints in the image
        """

        wpts_list =  [(20, 200), (200, 200)]
        chosen_wpt = []
        radius = 4

        # For now, let's just plot wpts onto the image
        for wpts in wpts_list:
            rgb_image = cv2.circle(rgb_image, wpts, radius=radius, color=(0, 0, 255), thickness=1)


        def draw_circle(event,x,y,flags,param):
            global mouseX,mouseY
            if event == cv2.EVENT_LBUTTONDBLCLK:
                mouseX,mouseY = x,y
                # TODO: check which waypoint has been clicked

        cv2.namedWindow('image')
        cv2.setMouseCallback('image',draw_circle)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', rgb_image)
        k = cv2.waitKey(20) & 0xFF

finally:

    # Stop streaming
    pipeline.stop()
