#! /usr/bin/env python

import time
import numpy as np
import peggnet
import torch.utils.data  
import sys
import cv2
import scipy.ndimage as ndimage
from skimage.draw import circle
from skimage.feature import peak_local_max
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray

bridge = CvBridge()

# Load Network
MODEL_FILE = './output/pre-trained_models/220206_0959_peggnet_jacquard_rgbd_480/epoch_45_iou_0.93_statedict.pt'

net = peggnet.PEGG_NET(input_channels=4)
net.load_state_dict(torch.load(MODEL_FILE))
net.eval()
device = torch.device("cuda:0")
net = net.to(device)

rospy.init_node('peggnet_rgbd_detection')

# Output publishers.
grasp_pub = rospy.Publisher('peggnet/img/grasp', Image, queue_size=1)
grasp_plain_pub = rospy.Publisher('peggnet/img/grasp_plain', Image, queue_size=1)
depth_pub = rospy.Publisher('peggnet/img/depth', Image, queue_size=1)
ang_pub = rospy.Publisher('peggnet/img/ang', Image, queue_size=1)
cmd_pub = rospy.Publisher('peggnet/out/command', Float32MultiArray, queue_size=1)

# Initialise some globals.
prev_mp = np.array([150, 150])
ROBOT_Z = 0

# Define image size
crop_size = 480
reshape_size = 480 

# Get the camera parameters
camera_info_msg = rospy.wait_for_message('/camera/aligned_depth_to_color/camera_info', CameraInfo)
K = camera_info_msg.K
fx = K[0]
cx = K[2]
fy = K[4]
cy = K[5]



def depth_callback(depth_message):
    start = time.time()
    global model
    global graph
    global prev_mp
    global fx, cx, fy, cy
    global COLOR
    depth = bridge.imgmsg_to_cv2(depth_message)
    depth_copy = depth
    depth_crop = cv2.resize(depth[(480-crop_size)//2:(480-crop_size)//2+crop_size, (640-crop_size)//2:(640-crop_size)//2+crop_size], (reshape_size, reshape_size))
    depth_crop = depth_crop.copy()
    depth_nan = np.isnan(depth_crop).copy()
    depth_crop[depth_nan] = 0

    # open cv inpainting does weird things at the border.
    depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)

    mask = (depth_crop == 0).astype(np.uint8)
    depth_scale = np.abs(depth_crop).max()
    depth_crop = depth_crop.astype(np.float32)/depth_scale 

    depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS)

    depth_crop = depth_crop[1:-1, 1:-1]
    depth_crop = depth_crop * depth_scale
    depth_center = depth_crop[100:141, 130:171].flatten()
    depth_center.sort()
    depth_center = depth_center[:10].mean() * 1000.0


    # Normalization for depth image captured by RealSense L515 camera.
    depth_input = depth_crop.copy()
    mean = depth_input.mean()
    depth_input = depth_input/mean
    depth_input -=1.3
    depth_input/=2.5

    #Concatenate depth and rgb image
    rgb_input = COLOR
    rgb_input = rgb_input.astype(np.float32)/255.0
    rgb_input = rgb_input-rgb_input.mean()
    depth_crop = depth_crop.reshape(reshape_size,reshape_size,1)
    depth_input = depth_input.reshape(reshape_size,reshape_size,1)
    color_input = torch.tensor(np.transpose(rgb_input,(2,0,1)))
    depth_input = torch.tensor(np.transpose(depth_input,(2,0,1)))
    depth_input = torch.cat((depth_input,color_input),0)
    depth_input = torch.unsqueeze(depth_input,0).to(device,dtype=torch.float32)

    with torch.no_grad():
        pred_out = net(depth_input)
    points_out = pred_out[0].squeeze()
    points_out[depth_nan] = 0


    # Calculate the angle map.
    cos_out = pred_out[1].squeeze()
    sin_out = pred_out[2].squeeze()
    ang_out = np.arctan2(sin_out, cos_out)/2.0
    width_out = pred_out[3].squeeze() * 150.0

    # Filter the outputs.
    points_out = ndimage.filters.gaussian_filter(points_out, 1) 
    ang_out = ndimage.filters.gaussian_filter(ang_out, 2.0)
    maxes = None
    ALWAYS_MAX = False  # Use ALWAYS_MAX = True for the open-loop solution.
    if ALWAYS_MAX:
        max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
        prev_mp = max_pixel.astype(np.int)
    else:
        # Calculate a set of local maxes.  Choose the one that is closes to the previous one.
        maxes = peak_local_max(points_out, min_distance=10, threshold_abs=0.1, num_peaks=5)
        if maxes.shape[0] == 0:
            return
        max_pixel = maxes[np.argmin(np.linalg.norm(maxes - prev_mp, axis=1))]

        # Keep a global copy for next iteration.
        prev_mp = (max_pixel * 0.25 + prev_mp * 0.75).astype(np.int)
    ang = ang_out[max_pixel[0], max_pixel[1]]
    width = width_out[max_pixel[0], max_pixel[1]]
    # Convert max_pixel back to uncropped/resized image coordinates in order to do the camera transform.
    max_pixel = ((np.array(max_pixel) / float(reshape_size) * crop_size) + np.array([(480 - crop_size)//2, (640 - crop_size) // 2]))
    max_pixel = np.round(max_pixel).astype(np.int)
    point_depth = depth_copy[max_pixel[0], max_pixel[1]]/1000
    x = (max_pixel[1] - cx)/(fx) * point_depth
    y = (max_pixel[0] - cy)/(fy) * point_depth
    z = point_depth

    # Visualization of grasp rectangle
    grasp_img = np.zeros((reshape_size, reshape_size, 3), dtype=np.uint8)
    grasp_img[:,:,2] = (points_out * 255.0)
    grasp_img_plain = grasp_img.copy()
    rr, cc = circle(prev_mp[0], prev_mp[1], 5)
    grasp_img[rr, cc, 0] = 0
    grasp_img[rr, cc, 1] = 255
    grasp_img[rr, cc, 2] = 0
    x0 = int(prev_mp[0])
    y0 = int(prev_mp[1])
    l_width = width
    dx = np.sin(ang) * width/2
    dy = np.cos(ang) * width/2
    x1 = int(x0 + dx)
    y1 = int(y0 - dy)
    
    x2 = int(x0 - dx)
    y2 = int(y0+dy)
    ang_2 = ang+1.57
    h = 30
    dx2 = np.sin(ang_2) * h /2
    dy2 = np.cos(ang_2) * h /2
    x3 = int(x1-dx2)
    y3 = int(y1+dy2)
    x4 = int(x1+dx2)
    y4 = int(y1-dy2)

    x5 = int(x2-dx2)
    y5 = int(y2+dy2)
    x6 = int(x2+dx2)
    y6 = int(y2-dy2)
    grasp_img = cv2.line(grasp_img, [y3,x3], [y4,x4], (255,255,0),2)
    grasp_img = cv2.line(grasp_img, [y5,x5], [y6,x6], (255,255,0),2)
    grasp_img = cv2.line(grasp_img, [y3,x3], [y5,x5], (0,255,255),2)
    grasp_img = cv2.line(grasp_img, [y4,x4], [y6,x6], (0,255,255),2)
    # Publish the output images (only for visualisation)
    grasp_img = bridge.cv2_to_imgmsg(grasp_img, 'bgr8')
    grasp_img.header = depth_message.header
    grasp_pub.publish(grasp_img)
    grasp_img_plain = bridge.cv2_to_imgmsg(grasp_img_plain, 'bgr8')
    grasp_img_plain.header = depth_message.header
    grasp_plain_pub.publish(grasp_img_plain)
    depth_pub.publish(bridge.cv2_to_imgmsg(np.array(depth_crop)))
    ang_pub.publish(bridge.cv2_to_imgmsg(ang_out))
    # Output the best grasp pose relative to camera.
    cmd_msg = Float32MultiArray()
    cmd_msg.data = [x, y, z, ang, width, depth_center]
    cmd_pub.publish(cmd_msg)

def color_callback(depth_message):
    start = time.time()
    global model
    global graph
    global prev_mp
    global ROBOT_Z
    global fx, cx, fy, cy
    global DEPTHIMG
    global COLOR

    rgb_raw = bridge.imgmsg_to_cv2(depth_message,"bgr8")
    rgb_crop = rgb_raw
    rgb_crop = cv2.resize(rgb_crop,(640,480))
    rgb_crop = rgb_crop[(480-crop_size)//2:(480-crop_size)//2+crop_size, (640-crop_size)//2:(640-crop_size)//2+crop_size,:]
    COLOR = rgb_crop



color_sub = rospy.Subscriber('/camera/color/image_raw', Image, color_callback, queue_size=1)
depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_callback, queue_size=1)

while not rospy.is_shutdown():
    rospy.spin()
