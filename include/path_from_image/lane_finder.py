#!/usr/bin/env python3
import rospy

import numpy as np
import cv2
from scipy.signal import find_peaks
        
def drawLaneArea(cv_image, transformMatrix, inverseMatrix):
    """Draw lane area on top of the image
        find points of the middle line of the lane
    
    Args:
        cv_image (OpenCV image): image to be drawn on
    
    Returns:
        np.array: image with lane area drawn on it
    """        
    warp_img = warp(cv_image, transformMatrix, inverseMatrix)
    binary = treshold_binary(warp_img)
    # treshold_img = np.dstack((binary, binary, binary)) * 255
    left_fit, right_fit, out_img = fit_polynomial(binary)
    draw_img = draw_filled_polygon(out_img, left_fit, right_fit)
    waypoints, draw_img = get_middle_line(draw_img, left_fit, right_fit, draw=True)
    unwarped_waypoints = cv2.perspectiveTransform(np.array([waypoints]), inverseMatrix)
    
    unwarp_img = warp(draw_img, transformMatrix, inverseMatrix, top_view=False)
    lane_area_img = cv2.addWeighted(cv_image,  0.8, unwarp_img,  0.7, 0)
    # lane_area_img = cv2.addWeighted(warp_img,  0.8, draw_img,  0.7, 0)
    
    return lane_area_img, unwarped_waypoints

def warp(image, transformMatrix, inverseMatrix, top_view=True):      
    """wrap image into top-view perspective

    Args:
        image (cv2_img): image to transform
        top_view (bool, optional): warp into top-view perspective or vice versa. Defaults to True.

    Returns:
        cv2_img: warped image
    """        
    h, w = image.shape[0], image.shape[1]
    birds_image = image
    matrix = transformMatrix
    if not top_view:
        matrix = inverseMatrix
    if (matrix is None):
        print("before warp call set_transform_matrix()")
        return birds_image
    else:
        birds_image = cv2.warpPerspective(np.copy(image), matrix, (w, h))            
    return birds_image

def treshold_binary(image, s_thresh=(90, 255), sx_thresh=(10, 100)):
    """Gradient and color tresholding of an image 

    Args:
        image (OpenCV image): image to be tresholded
        s_thresh (tuple, optional): _description_. Defaults to (50, 255).
        sx_thresh (tuple, optional): _description_. Defaults to (20, 100).

    Returns:
        np.array: binary image
    """        
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
        
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx)) if (np.max(abs_sobelx)) else np.uint8(255*abs_sobelx)
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    # in simulation h and s channels are zeros
    if (np.max(s_channel) == 0):
        s_channel = l_channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def fit_polynomial(treshold_warped, ym_per_pix=1, xm_per_pix=1):
    """return polynomial for both lanes

    Args:
        treshold_warped (CVimage): tresholded binary image
        ym_per_pix (int, optional): meters per pixel in y dimension. Defaults to 1.
        xm_per_pix (int, optional): meters per pixel in x dimension. Defaults to 1.

    Returns:
        tuple: polynomials for right and left lane
    """        
    
    left_fit = [0,0,0]
    right_fit = [0,0,0]
        
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(treshold_warped)
    
    # check if the image is not empty
    if (leftx.size > 0 and lefty.size > 0 and rightx.size > 0 and righty.size > 0):
        left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)      
    
    return left_fit, right_fit, out_img

def find_lane_pixels(image, draw=False):
    """find lane pixels in image with sliding windows

    Args:
        image (CVimage): tresholded binary image
        draw (bool, optional): draw rectangles on the image or not. Defaults to False.

    Returns:
        tuple: right and left lane pixels
    """    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 20
    # Set the width of the windows +/- margin
    margin = 40
    # Set minimum number of pixels found to recenter window
    minpix = 100

    # Take a histogram of the bottom half of the image
    histogram = np.sum(image[image.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((image, image, image))
    peaks, _ = find_peaks(histogram, distance=100)
    # if the image is empty
    if (np.amax(histogram) < 10):
        leftx = lefty = rightx = righty = np.array([])
    else:        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int64(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint-margin])
        rightx_base = np.argmax(histogram[midpoint+margin:]) + midpoint  
        leftx_base = peaks[0]
        rightx_base = peaks[-1]

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int64(image.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window+1)*window_height
            win_y_high = image.shape[0] - window*window_height
            
            # Find the four below boundaries of the window 
            win_xleft_low = leftx_current - margin 
            win_xleft_high = leftx_current + margin                     
            # draw window rectangles
            if draw:
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                (win_xleft_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window 
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            # If you found > minpix pixels, recenter next window on their mean position 
            # or stop windows
            if len(good_left_inds) > minpix:
                leftx_current=np.int64(np.mean(nonzerox[good_left_inds]))

            # Find the four below boundaries of the window 
            win_xright_low = rightx_current - margin 
            win_xright_high = rightx_current + margin                      
            # draw window rectangles
            if draw:
                cv2.rectangle(out_img,(win_xright_low,win_y_low),
                (win_xright_high,win_y_high),(0,255,0), 2)  

            # Identify the nonzero pixels in x and y within the window                     
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position 
            # or stop windows
            if (good_right_inds.shape[0] > minpix):
                rightx_current=np.int64(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        out_img[lefty, leftx] = [0, 0, 255]
        out_img[righty, rightx] = [255, 0, 0]

    return leftx, lefty, rightx, righty, out_img 

def draw_filled_polygon(img, left_fit, right_fit):
    """draw filled polygon on the image

    Args:
        img (CVimage): image to be drawn on
        left_fit (_type_): left lane line
        right_fit (_type_): right lane line

    Returns:
        CVimage: image with filled polygon
    """        
    draw_img = np.copy(img)
    ploty, left_fitx, right_fitx = get_xy(draw_img.shape[0], draw_img.shape[0], left_fit, right_fit)
    
    all_x = np.concatenate([left_fitx, np.flip(right_fitx, 0)])
    all_y = np.concatenate([ploty, np.flip(ploty, 0)])
    all_points = [(np.asarray([all_x, all_y]).T).astype(np.int32)]
    cv2.fillPoly(draw_img, all_points, (0,255,0))
    
    return draw_img

def get_xy(shape_0, num_points, left_fit, right_fit):
    """get x and y coordinates of the lane area

    Args:
        shape_0 (int): height of the image
        left_fit (array): left lane line
        right_fit (array): right lane line

    Returns:
        tuple: y coordinates and left and right x coordinates
    """        
    ploty = np.linspace(0, shape_0-1, num_points )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
        
    return ploty, left_fitx, right_fitx

def get_middle_line(img, left_fit, right_fit, draw=False):
    """get middle line of the lane area

    Args:
        img (CVimage): image to be drawn on
        left_fit (_type_): left lane line
        right_fit (_type_): right lane line
        draw (bool, optional): Draw markers on the image. Defaults to False.

    Returns:
        tuple: middle line and image with markers
    """        
    draw_img = np.copy(img)
    # get only 10 points, it is enough
    ploty, left_fitx, right_fitx = get_xy(img.shape[0], 10, left_fit, right_fit)
    middle_fitx = (left_fitx + right_fitx) / 2
    middle_lane_points = np.asarray([middle_fitx, ploty]).T
    if draw:
        for point in middle_lane_points.astype(np.int64):
            cv2.drawMarker(draw_img, (point[0], point[1]), (255, 255, 255), cv2.MARKER_CROSS, thickness=5)

    return middle_lane_points, draw_img