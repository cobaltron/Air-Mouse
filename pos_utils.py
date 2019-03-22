import numpy as np
import cv2
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):
   # compute the euclidean distances between the two sets of
   # vertical eye landmarks (x, y)-coordinates
  A = dist.euclidean(eye[1], eye[5])
  B = dist.euclidean(eye[2], eye[4])
   # compute the euclidean distance between the horizontal
   # eye landmark (x, y)-coordinates
  C = dist.euclidean(eye[0], eye[3])
   # compute the eye aspect ratio
  ear = (A + B) / (2.0 * C)
   # return the eye aspect ratio
  return ear


def imshow(img):
    cv2.imshow('frame',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pose(im,n,c,le,re,lm,rm):
    size = im.shape
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                (n[0], n[1]),     # Nose tip
                                (c[0], c[1]),     # Chin
                                (le[0], le[1]),     # Left eye left corner
                                (re[0], re[1]),     # Right eye right corne
                                (lm[0], lm[1]),     # Left Mouth corner
                                (rm[0], rm[1])      # Right mouth corner
                            ], dtype="double")

    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner

                            ])
    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )

    #print("Camera Matrix :\n {0}".format(camera_matrix))

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    '''
    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
    '''
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    cv2.line(im, p1, p2, (255,0,0), 2)
    # Display image
    #cv2.imshow("output",im)
    #cv2.waitKey(10)
    return p2,im
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords



