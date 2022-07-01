#!/usr/bin/env python
import numpy as np
import cv2
from sympy import Point3D, Line3D, Plane

from image_geometry import PinholeCameraModel


# class uses cameraInfo and 3d geometry to warp camera images
class topViewer:
    def __init__(self):
        # camera model
        self.camera_model = PinholeCameraModel()
        # transformation matrix for 
        # (un)wraping images to top-view projection
        self.transformMatrix = None
        # top-view image
        self.birds_image = None
        # (x_scale, y_scale) - scale factor in m per pixel
        self.scale = None

    def __call__(self, image):        
        if (self.transformMatrix is not None):
            # warp it to the top-view projection
            self.warp_perspective(image)            
        return self.birds_image

    def setScale(self, scale):
        self.scale = scale

    def get_transform_matrix(self, src, dst):
        # get matrix for perspective transformation
        self.transformMatrix = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst)) 
        # TODO do we need it here?
        # self.setScale((x_scale/w, y_scale/h))

    # get_transform_matrix(src, dst) should be called before
    def warp_perspective(self, img):
        h, w = img.shape[0], img.shape[1]
        if (self.transformMatrix is None):
            print("before warp call get_transform_matrix()")
        else:
            self.birds_image = cv2.warpPerspective(np.copy(img), self.transformMatrix, (w, h))            
        return self.birds_image

    # get upper src points in the base_link frame
    # transformed points are used in 
    # ImageProcessor.get_transform_matrix(src, dst)
    # for (un)wrapping image into top-view perspective.
    # All points are in the base_link frame
    # zero - camera center
    # lbc - ray which goes from center and left bottom corner of the image
    # rbc - ray which goes from center and right bottom corner of the image
    # xoy - any point which lay on the ground
    # y_scale - scale factor, meters per image height
    def getUpperPoints(self, zero, lbc, rbc, y_scale=2, xoy=(0, 0, 0)):
        # make two lines from camera center
        lbc_line = Line3D(Point3D(zero), Point3D(lbc))
        rbc_line = Line3D(Point3D(zero), Point3D(rbc))
        # ground plane with lanes
        xy_plane = Plane(Point3D(xoy), normal_vector=(0, 0, 1))
        # bottom points in the base_bottom_link frame
        # are intersection points of lines and ground plane
        point1 = xy_plane.intersection(lbc_line)[0]
        point2 = xy_plane.intersection(rbc_line)[0]
        # translate factor in meters
        # depends on h*w of the image
        # and scale factor
        x_scale = float(point1.distance(point2))
        # upper points in the base_link frame
        point3 = point1.translate(y_scale)
        point4 = point2.translate(y_scale)
        return point3, point4, x_scale