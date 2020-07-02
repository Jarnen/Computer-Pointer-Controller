
import math
import numpy as np
import cv2

# ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def draw_axes(frame, yaw, pitch, roll, scale, focal_length):
    center_of_face = get_center(frame)
    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                   [0, 1, 0],
                   [math.sin(yaw), 0, math.cos(yaw)]])
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll), math.cos(roll), 0],
                   [0, 0, 1]])
    # R = np.dot(Rz, Ry, Rx)
    # R = np.dot(Rz, np.dot(Ry, Rx))
    R = Rz @ Ry @ Rx
    
    camera_matrix = build_camera_matrix(center_of_face, focal_length)
    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)

    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]

    xaxis = np.dot(R, xaxis) + o
    yaxis = np.dot(R, yaxis) + o
    zaxis = np.dot(R, zaxis) + o
    zaxis1 = np.dot(R, zaxis1) + o
    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)

    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)

    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))

    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, p1, p2, (255, 0, 0), 2)

    return frame

#ref: http://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf

def build_camera_matrix(camera_center, focal_length):
    """
    Returns camera projection matrix

    Args
    ------------------
        camera_center:  tuple (x,y) of the center coordinates
        focal_lenght:   approximate focal length of the camera in mm
    """
    f = focal_length
    cx = camera_center[0]
    cy = camera_center[1]

    matrix = np.array([[f, 0, cx],
                       [0, f, cy],
                       [0, 0, 1]])
    return matrix
    
#ref: https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python

def get_center(frame):
    # # convert image to grayscale image
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # # convert the grayscale image to binary image
    # ret,thresh = cv2.threshold(gray_frame,127,255,0)
    
    # # calculate moments of binary image
    # M = cv2.moments(thresh)
    # # calculate x,y coordinate of center
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])
    height, width = frame.shape[0:2]
    cX = width/2
    cY = height/2

    return (cX, cY)