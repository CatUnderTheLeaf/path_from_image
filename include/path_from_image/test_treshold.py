import os
import cv2
import numpy as np
import lane_finder
from keras.models import load_model

# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """

    # Get image ready for feeding into model
    # size = (160,80)
    # small_img = cv2.resize(image[image.shape[0]//2-20:,:], size)
    small_img = np.array(image)
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
    blanks = np.zeros_like(prediction).astype(np.uint8)
    # lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))
    lane_drawn = np.dstack((blanks, blanks, prediction))
    # Re-size to match the original image
    # lane_image = cv2.resize(lane_drawn, (410, 308))

    print(image.shape)
    print(lane_drawn.shape)

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image.astype(np.uint8), 0.8, lane_drawn.astype(np.uint8), 1, 0)
    # result = lane_drawn
    return result

transform_matrix = np.array(
                            # [[-1.44070347e+00, -2.24787583e+00,  5.03003158e+02],
                            # [ 5.54569940e-16, -4.76928443e+00,  7.40727232e+02],
                            # [ 1.37242848e-18, -1.09231360e-02,  1.00000000e+00]])
                             [[-8.45065754e-01, -2.25563747e+00,  3.80835430e+02],
                              [ 8.86394010e-03, -4.21709202e+00,  5.64499885e+02],
                               [ 2.87790263e-05, -1.09879876e-02,  1.00000000e+00]])
inverse_matrix =np.array(
        	            # [[4.83440511e-01, -4.72483945e-01,  1.06809629e+02],
                        # [ 1.49479041e-16, -2.09675060e-01,  1.55312027e+02],
                        # [ 9.62443421e-19, -2.29030920e-03,  1.00000000e+00]])
                        [ [5.54069331e-01, -5.38262142e-01,  9.28396978e+01],
                          [2.05982340e-03,  -2.38865552e-01,  1.34055128e+02],
                            [6.68773796e-06, -2.60916114e-03,  1.00000000e+00]])
camera_mtx = np.array([[202.35126146, 0, 206.80143037], [0, 201.08542928, 153.32497517], [0, 0, 1]])
dist = np.array([-3.28296006e-01, 1.19443754e-01, -1.85799276e-04, 8.39998127e-04, -2.06502314e-02] )

test_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # print(test_dir_path)
camera_img = cv2.imread(os.path.join(test_dir_path, 'resource', 'frame0008.jpg'))
# undistort if not already
undist = cv2.undistort(camera_img, camera_mtx, dist, None, camera_mtx)
lane_image = cv2.resize(camera_img, (205, 154))
cropped = lane_image[1:lane_image.shape[0]-1, 2:lane_image.shape[1]-3]

model_path = os.path.join(test_dir_path, 'full_conv_network', 'FCNN_model.h5')
model = load_model(model_path)
# Create lanes object
lanes = Lanes()

resized_image = road_lines(cropped)
cv2.imwrite(os.path.join(test_dir_path, 'resource', 'lane_image.jpg'), resized_image)



# import glob
# # resize images
# img_path = os.path.join(test_dir_path, 'full_conv_network', 'dataset', 'train', '*')
# images = glob.glob(img_path)
# i = 4119
# for fname in images:
#     img = cv2.imread(fname)
#     # res_image = cv2.resize(img, (205, 154))
#     res_image = cv2.flip(img, 1)
#     head_tail = os.path.split(fname)
#     name = 'frame'+ str(i)+'.jpg'
#     # name = head_tail[1]
#     print(name)
#     cv2.imwrite(os.path.join(test_dir_path, 'full_conv_network', 'dataset', 'temp', name), res_image)
#     i+=1

# make line images
# img_path = os.path.join(test_dir_path, 'bagfiles', 'train', '*')
# images = glob.glob(img_path)
# for fname in images:
#     img = cv2.imread(fname)
#     lane_img, points = lane_finder.drawMiddleLine(img)
#     b_channel = lane_img[:,:,1]
#     b_channel = b_channel[:, :, None]
#     head_tail = os.path.split(fname)
#     cv2.imwrite(os.path.join(test_dir_path, 'bagfiles', 'labels', head_tail[1]), b_channel)
#     # cv2.imwrite(os.path.join(test_dir_path, 'bagfiles', 'labels', head_tail[1]), lane_img)