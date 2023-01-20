#!/usr/bin/env python3
import rospy
import os
import numpy as np
import cv2
from scipy.signal import find_peaks
from keras.models import load_model

package_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(package_dir_path, 'full_conv_network', 'FCNN_model.h5')
model = load_model(model_path)
# Class to average lanes with
# class Lanes():
#     def __init__(self):
#         self.recent_fit = []
#         self.avg_fit = []

def drawMiddleLine(cv_image, useFCNN=True):
    """Draw line on top of the image
        get several points of it
    
    Args:
        cv_image (OpenCV image): image to be drawn on
    
    Returns:
        np.array: image with line drawn on it
        np.array(x,y): a couple of points of the middle line
    """    
    if (useFCNN):
        binary = predict_binary(cv_image)
    else:
        binary = yellow_treshold_binary(cv_image)
    pixels_line_img, nonzerox, nonzeroy = find_middle_line_pixels(binary)
    
    # return pixels_line_img, np.array(cv_image)
    if (nonzerox.size and nonzeroy.size):
        middle_fit = get_polynomial(nonzeroy, nonzerox)
        line_img, lane_points = draw_polyline(cv_image, middle_fit)
        # line_img, lane_points = draw_polyline(np.zeros_like(cv_image), middle_fit)
        return line_img, np.array(lane_points)
    else:
        return cv_image, []

def predict_binary(cv_image):
    small_img = np.array(cv_image)
    small_img = cv2.resize(small_img, (205, 154))
    small_img = small_img[1:small_img.shape[0]-1, 2:small_img.shape[1]-3]
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    # lanes.recent_fit.append(prediction)
    # # # Only using last five for average
    # if len(lanes.recent_fit) > 5:
    #     lanes.recent_fit = lanes.recent_fit[1:]

    # # Calculate average detection
    # lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # # Generate fake R & B color dimensions, stack with G
    # blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    # blanks = np.zeros_like(prediction).astype(np.uint8)
    # lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))
    # lane_drawn = np.dstack((blanks, blanks, prediction))
    # Re-size to match the original image
    # lane_image = cv2.resize(lane_drawn, (410, 308))

    # print(image.shape)
    # print(lane_drawn.shape)

    # Merge the lane drawing onto the original image
    # result = cv2.addWeighted(image.astype(np.uint8), 0.8, lane_drawn.astype(np.uint8), 1, 0)
    # result = lane_drawn

    # return to initial size
    prediction = prediction[:,:,0]
    
    result = np.full((cv_image.shape[0]//2, cv_image.shape[1]//2), 0, dtype=np.uint8)
    result[1:cv_image.shape[0]//2-1, 2:cv_image.shape[1]//2-3] = prediction
    result = cv2.resize(result, (cv_image.shape[1], cv_image.shape[0]))
     
    return result

def drawLaneArea(cv_image, transformMatrix, inverseMatrix):
    """Draw lane area on top of the image
        find points of the middle line of the lane
    
    Args:
        cv_image (OpenCV image): image to be drawn on
    
    Returns:
        np.array: image with lane area drawn on it

    """      
    warp_img = warp(cv_image, transformMatrix, inverseMatrix)
    # return warp_img, np.array(cv_image)  
    binary = treshold_binary(warp_img)
    # binary = treshold_binary(cv_image)    
    treshold_img = np.dstack((binary, binary, binary)) * 255
    # return treshold_img, np.array(cv_image)  

    left_fit, right_fit, out_img = fit_polynomial(binary)
    # return out_img, np.array(cv_image)  
    draw_img = draw_filled_polygon(out_img, left_fit, right_fit)
    # return draw_img, np.array(cv_image)  
    
    waypoints, draw_img = get_middle_line(draw_img, left_fit, right_fit, draw=True)
    # return draw_img, waypoints
    unwarped_waypoints = cv2.perspectiveTransform(np.array([waypoints]), inverseMatrix)
    # return draw_img, unwarped_waypoints
    unwarp_img = warp(draw_img, transformMatrix, inverseMatrix, top_view=False)
    # return unwarp_img, unwarped_waypoints
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
    matrix = transformMatrix
    if not top_view:
        matrix = inverseMatrix
    if (matrix is None):
        print("before warp call set_transform_matrix()")
        return image
    else:
        birds_image = cv2.warpPerspective(image, matrix, (w, h))            
    return birds_image

def treshold_binary(image, s_thresh=(130, 190), sx_thresh=(30, 100)):
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

def yellow_treshold_binary(image):
    """yellow color tresholding of an image,
    upper part of image is cut, it doesn't have relevant information

    Args:
        image (OpenCV image): image to be tresholded
    Returns:
        np.array: binary image
    """        
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_thresh=(140, 255)
    # b channel for yellow and blue colors
    b_channel = lab[:,:,2]
    lab_binary = np.zeros_like(b_channel)
    lab_binary[(b_channel >= lab_thresh[0]) & (b_channel <= lab_thresh[1])] = 1
    
    
    # yellow_output[(th > 0)] = 255
    s_thresh=(30, 50)
    sx_thresh=(10, 100) #20, 100
    # Convert to HLS color space
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
    
    # Threshold light channel
    # in simulation h and s channels are zeros
    # if (np.max(s_channel) == 0):
    #     s_channel = l_channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Combine the two binary thresholds
    yellow_output = np.zeros_like(sxbinary)
    # this outlines 3 lines
    yellow_output[(sxbinary == 1) & (lab_binary == 1) & (s_binary != 1)] = 1
    # yellow_output[(sxbinary == 1) & (lab_binary == 1)] = 1
    # yellow_output[(lab_binary == 1)] = 1
    # yellow_output[(sxbinary == 1)] = 1
    # yellow_output[(s_binary == 1)] = 1
    # yellow_output[(h_binary == 1)] = 1
    
    yellow_output[:yellow_output.shape[0]//2-15, :yellow_output.shape[1]] = 0
    return yellow_output

def fit_polynomial(treshold_warped):
    """find lane pixels and return polynomial for both lanes

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
        left_fit = get_polynomial(lefty, leftx)
        right_fit = get_polynomial(righty, rightx)      
    
    return left_fit, right_fit, out_img

def get_polynomial(y, x, ym_per_pix=1, xm_per_pix=1):
    """polynomial for a single line

    Args:
        y(array): y-coordinates of a line
        x(array): x-coordinates of a line
        ym_per_pix (int, optional): meters per pixel in y dimension. Defaults to 1.
        xm_per_pix (int, optional): meters per pixel in x dimension. Defaults to 1.

    Returns:
        array: polynomial of the line
    """  
    return np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)

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

    # # Take a histogram of the bottom half of the image
    histogram = np.sum(image[image.shape[0]//2:,:], axis=0)
    # # Create an output image to draw on and visualize the result
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
    # leftx = lefty = rightx = righty = np.array([])
    return leftx, lefty, rightx, righty, out_img 

def find_middle_line_pixels(image, draw=False):
    """find middle line pixels in image with sliding windows

    Args:
        image (CVimage): tresholded binary image
        draw (bool, optional): draw rectangles on the image or not. Defaults to False.

    Returns:
        tuple: line nonzero pixels
    """    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 30
    # Set the width of the windows +/- margin
    margin = 35
    # Set minimum number of pixels found to recenter window
    minpix = 2

    # seen = False

    # # Take a histogram of the bottom half of the image
    histogram = np.sum(image[image.shape[0]//6*5:,:], axis=0)
    # # Create an output image to draw on and visualize the result
    out_img = np.dstack((image, image, image)) *255
    # return out_img, np.array([]), np.array([])
    peaks, properties = find_peaks(histogram, height =4, width=1)
    # import matplotlib.pyplot as plt
    # plt.plot(histogram)
    # plt.plot(peaks, histogram[peaks], "x")
    # plt.show()
    # print(properties)
    # print(properties['peak_heights'])
    # if the image is empty
    if (not len(properties['peak_heights'])):
        leftx = lefty = np.array([])
        # import os
        # test_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # name = str(np.random.randint(100))
        # cv2.imwrite(os.path.join(test_dir_path, 'resource', name+'.jpg'), orig)
    else:        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        point = np.argmax(properties['peak_heights'])
        leftx_base = peaks[point]
        # print(leftx_base)
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int64((image.shape[0]//2+35)//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window+1)*window_height
            win_y_high = image.shape[0]- window*window_height
            
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
            #     seen = True
            # if not len(good_left_inds) and not seen:
            #     import os
            #     test_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            #     name = str(np.random.randint(100))
            #     cv2.imwrite(os.path.join(test_dir_path, 'resource', name+'.jpg'), orig)
            #     seen = True
            
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 

        out_img[lefty, leftx] = [0, 0, 255]
    return  out_img, leftx, lefty

def draw_polyline(img, fit):
    """draw polyline on the image

    Args:
        img (CVimage): image to be drawn on
        fit (_type_): lane line
    Returns:
        CVimage: image with a polyline
    """        
    draw_img = np.copy(img)
    ploty = np.linspace(draw_img.shape[0]//2, draw_img.shape[0]-1, 10)
    fitx = get_xy(ploty, fit)
        
    all_points = [(np.asarray([fitx, ploty]).T).astype(np.int32)]
    cv2.polylines(draw_img, all_points, False, (0, 255, 0), 2)
    lane_points = np.asarray([fitx, ploty]).T
    
    return draw_img, lane_points

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
    ploty = np.linspace(0, draw_img.shape[0]-1, draw_img.shape[0])
    left_fitx = get_xy(ploty, left_fit)
    right_fitx = get_xy(ploty, right_fit)
    
    all_x = np.concatenate([left_fitx, np.flip(right_fitx, 0)])
    all_y = np.concatenate([ploty, np.flip(ploty, 0)])
    all_points = [(np.asarray([all_x, all_y]).T).astype(np.int32)]
    cv2.fillPoly(draw_img, all_points, (0,255,0))
    
    return draw_img

def get_xy(ploty, fit):
    """get x coordinates of the lane

    Args:
        ploty (array): y-coordinates
        fit (array): lane line

    Returns:
        tuple: y coordinates and left and right x coordinates
    """        
    try:
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        fitx = 1*ploty**2 + 1*ploty
        
    return fitx

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
    ploty = np.linspace(0, img.shape[0]-1, 10)
    left_fitx = get_xy(ploty, left_fit)
    right_fitx = get_xy(ploty, right_fit)
    middle_fitx = (left_fitx + right_fitx) / 2
    middle_lane_points = np.asarray([middle_fitx, ploty]).T
    if draw:
        for point in middle_lane_points.astype(np.int64):
            cv2.drawMarker(draw_img, (point[0], point[1]), (255, 255, 255), cv2.MARKER_CROSS, thickness=5)

    return middle_lane_points, draw_img