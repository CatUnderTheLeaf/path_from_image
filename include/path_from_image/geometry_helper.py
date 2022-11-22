#!/usr/bin/env python3
import rospy
import math

def getLinePlaneIntersection(zero, point, z=0.0):
    """Get intersection of line with plane parallel/equal to Oxy
    
    Args:
        zero (tuple): camera center point in robot/world frame
        point (tuple): point of the image robot/world frame
        z (float): z-value of the plane
    
    Returns:
        tuple: intersection point
    """    
    # the point in the robot/world frame
    # is an intersection point of a line and ground plane
    # Line formula in 3D with two points:
    # (x-x1)/(x2-x1) = (y-y1)/(y2-y1) = (z-z1)/(z2-z1)
    # our plane is Oxy plane, so z=0
    # and intersection point equals
    # x = -z1*(x2-x1)/(z2-z1) + x1
    # y = -z1*(y2-y1)/(z2-z1) + y1
    new_x = (z-zero[2])*(point[0]-zero[0])/(point[2]-zero[2]) + zero[0]
    new_y = (z-zero[2])*(point[1]-zero[1])/(point[2]-zero[2]) + zero[1]

    return new_x, new_y, z

def getDistance(point1, point2):
    """Get distance between points
    
    Args:
        point1 (tuple): point
        point2 (tuple): point
    
    Returns:
        float: distance
    """  
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)

def translatePoint(point, x=0, y=0, z=0):
    """Get new point from translating old point
    
    Args:
        point (tuple): point
        x (float): value to translate point in x-direction
        y (float): value to translate point in y-direction
        z (float): value to translate point in z-direction
    
    Returns:
        tuple: new point
    """ 
    return point[0]+x, point[1]+y, point[2]+z

def getUpperPoints(zero, lbc, rbc, distance_ahead):
    """ get upper src points in the robot/world frame

    Args:
        zero (tuple): camera center
        lbc (tuple): ray which goes from center and left bottom corner of the image
        rbc (tuple): ray which goes from center and right bottom corner of the image
        distance_ahead(float): distance ahead of the car

    Returns:
        tuple: two upper points and x_scale factor
    """        
    point1 = getLinePlaneIntersection(zero, lbc, lbc[2])
    point2 = getLinePlaneIntersection(zero, rbc, rbc[2])

    # distance in meters between bottom points
    # after their projection onto the ground plane
    x_scale = getDistance(point1, point2)

    # get upper points in the robot/world frame
    # sometimes in car models front camera 
    # looks in the negative axe direction
    # so our translation should has right sign
    # TODO maybe has to be some way to determine 
    # which axis goes along the car model and
    # if it has the same direction as z-axis of the cv2-image
    if lbc[1] > 0:
        sign = 1
    else:
        sign = -1
    point3 = translatePoint(point1, sign*distance_ahead)
    point4 = translatePoint(point2, sign*distance_ahead)
    
    return point3, point4, x_scale