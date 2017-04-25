import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import re

# 'percentage_binary_thresholding.ipynb'
def percentages_to_thresholds(img_gray, lower_percentage, upper_percentage):
    norm_hist = (np.cumsum(np.histogram(img_gray, 255, (0.0, 255.0))[0]) / \
                 img_gray.size).astype(np.float32)
    values = np.arange(0, 255)[((norm_hist >= lower_percentage) & \
                                (norm_hist <= upper_percentage))]
    
    return np.min(values), np.max(values)

# 'gradient_experiment.ipynb'
def get_binary_gradient(img, low_thresh=0.95, high_thresh=1.0):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)[:,:,1]
    
    gradient_x = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0))
    gradient_x = 255 * gradient_x / np.max(gradient_x)
    
    binary = np.zeros_like(img_gray).astype(np.bool8)
    low_thresh, high_thresh = percentages_to_thresholds(gradient_x, \
        low_thresh, high_thresh)
    
    binary[(gradient_x >= low_thresh) & (gradient_x <= high_thresh)] = True
    return binary

# 'color_scheme_experiment.ipynb'
def Sally(img):
    S = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)[:,:,2]
    A = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:,:,1]
    
    return np.sqrt( \
                S.astype(np.uint32) * \
                A.astype(np.uint32) \
                  ).astype(np.uint8)

# 'color_scheme_experiment.ipynb'
def get_binary_color(img, low_thresh = 0.9, high_thresh = 1.0):
    img_gray = Sally(img)
    
    low_thresh, high_thresh = percentages_to_thresholds(img_gray, \
        low_thresh, high_thresh)
    
    img_binary = np.zeros_like(img_gray).astype(np.bool8)
    img_binary[(img_gray > low_thresh) & (img_gray < high_thresh)] = True
    
    return img_binary

from moviepy.editor import VideoFileClip

temp = None
def process_image(img):
    binary_R = 255 * np.zeros(img.shape[:2]).astype(np.uint8)
    binary_G = 255 * get_binary_color(img).astype(np.uint8)
    binary_B = 255 * get_binary_gradient(img).astype(np.uint8)
    binary = np.dstack((binary_R, binary_G, binary_B))

    global temp
    if temp is None:
        temp = np.copy(binary)
        
    return binary

# The video processing and storing code
input_video = VideoFileClip("white.mp4")
output_video = input_video.fl_image(process_image)

output_filename = 'project_output_1.mp4'
output_video.write_videofile(output_filename, audio=False)
