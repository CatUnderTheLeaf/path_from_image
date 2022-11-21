import os
import cv2
import numpy as np
import lane_finder

transform_matrix = np.array([[-1.44070347e+00, -2.24787583e+00,  5.03003158e+02],
                            [ 5.54569940e-16, -4.76928443e+00,  7.40727232e+02],
                            [ 1.37242848e-18, -1.09231360e-02,  1.00000000e+00]])
inverse_matrix =np.array([[4.83440511e-01, -4.72483945e-01,  1.06809629e+02],
                        [ 1.49479041e-16, -2.09675060e-01,  1.55312027e+02],
                        [ 9.62443421e-19, -2.29030920e-03,  1.00000000e+00]])
camera_mtx = np.array([[202.35126146, 0, 206.80143037], [0, 201.08542928, 153.32497517], [0, 0, 1]])
dist = np.array([-3.28296006e-01, 1.19443754e-01, -1.85799276e-04, 8.39998127e-04, -2.06502314e-02] )

test_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(test_dir_path)
camera_img = cv2.imread(os.path.join(test_dir_path, 'resource', 'frame0000.jpg'))

undist = cv2.undistort(camera_img, camera_mtx, dist, None, camera_mtx)

lane_img = lane_finder.drawLaneArea(undist, transform_matrix, inverse_matrix)
# path_from_image.src.lane_finder.drawLaneArea(cv_image, transform_matrix, inverse_matrix)
cv2.imwrite(os.path.join(test_dir_path, 'resource', 'lane_image.jpg'), lane_img)
