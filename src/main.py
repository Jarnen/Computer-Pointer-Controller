from argparse import ArgumentParser
from utils import draw_axes
import logging as log
import sys
import cv2
import numpy as np
from input_feeder import InputFeeder
from mouse_controller import MouseController
from inference import Model
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from landmarks_detection import LandmarksDetection
from gaze_estimation import GazeEstimation
import time

DEVICE_TYPES = ['CPU', 'GPU', 'FPGA', 'MYRAID', 'HETERO']

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument('-i', '--input_type', metavar='PATH', default=None,
                                  help="(optional) Path to the input video " \
                                      "('cam' for camera, default)")
    parser.add_argument('-f', '--input_file', metavar='PATH', default=None,
                                   help="(optional) Path to the input file " \
                                       "(None for input file, default)")
    parser.add_argument('-m_fd', metavar="PATH", default="", required=True,
                        help="Path to the Face Detection model XML file")
    parser.add_argument('-m_ld', metavar="PATH", default="", required=True,
                        help="Path to the Landmarks Detection model XML file")
    parser.add_argument('-m_pe', metavar="PATH", default="", required=True,
                        help="Path to the Pose Estimation model XML file")
    parser.add_argument('-m_ge', metavar="PATH", default="", required=True,
                        help="Path to the Gaze Estimation model XML file")
    
    parser.add_argument('-d', '--device', default='CPU', choices=DEVICE_TYPES,
                       help="(optional) Target device for the " \
                       "Face Detection model (default: %(default)s)")
    parser.add_argument('-l', '--lib', metavar="PATH", default="",
                       help="(optional) For targeted device custom layers, if any. " \
                       "Path to a shared library with custom layers implementations")
    parser.add_argument('-t','--threshold', metavar='[0..1]', type=float, default=0.5,
                       help="(optional) Probability threshold for face detections" \
                       "(default: %(default)s)")
    parser.add_argument('-v', '--verbose', action='store_true',
                       help="(optional) Be more verbose")
    
    return parser

class Visualize:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord('q'), ord('Q'), 27}

    def __init__(self, args):
        """
        Initialise the variables
        
        """
        self.face_detector = FaceDetection(args.m_fd, args.device, args.lib, args.threshold)
        self.landmarks_detector = LandmarksDetection(args.m_ld, args.device, args.lib, args.threshold)
        self.head_pose_estimation = HeadPoseEstimation(args.m_pe, args.device, args.threshold)
        self.gaze_estimation = GazeEstimation(args.m_ge,args.device, args.threshold)
        self.mouse_controller = MouseController('high', 'fast')

        self.face_detector.load_model()
        self.landmarks_detector.load_model()
        self.head_pose_estimation.load_model()
        self.gaze_estimation.load_model()

        self.feed = InputFeeder(args.input_type, args.input_file)
            
    def process(self, frame):
        assert len(frame.shape) == 3, "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], "Expected BGR or BGRA input"

        rois = self.face_detector.predict(frame)
        face = self.face_detector.preprocess_output(frame, rois)
        
        landmarks = self.landmarks_detector.predict(face)
        right_eye_image, left_eye_image, right_eye_roi, left_eye_roi = self.landmarks_detector.preprocess_output(face, landmarks)
        head_pose_angles = self.head_pose_estimation.predict(face)
        
        self.draw_pose_detection(face, head_pose_angles)
        self.draw_eye_landmarks(face, right_eye_roi, left_eye_roi)
        frame = self.draw_face_roi(frame, rois)
        
        x, y = self.gaze_estimation.predict(right_eye_image, left_eye_image, head_pose_angles)
        self.mouse_controller.move(x, y)

        self.display_window(frame)
        
    def draw_face_roi(self, frame, roi):
        """
        Draw the face ROI border
        """
        frame = cv2.rectangle(frame, (roi[0][0], roi[0][1]), (roi[0][2], roi[0][3]), (0,220,0) , 2)

        return frame
    
    def draw_eye_landmarks(self, face, right_eye, left_eye):
        """
        Draw the eyes ROI landmarks on the face
        """
        face = cv2.rectangle(face, (right_eye[0], right_eye[1]), (right_eye[2], right_eye[3]), (0,220,0) , 2)
        face = cv2.rectangle(face, (left_eye[0], left_eye[1]), (left_eye[2], left_eye[3]), (0,220,0) , 2)
    
    def draw_pose_detection(self, face, head_pose_angles): 
        """
        Draw yaw, pitch, roll angle lines for visualisation
        """
        scale = 60
        yaw = head_pose_angles[0][0]
        pitch = head_pose_angles[0][1]
        roll = head_pose_angles[0][2]

        draw_axes(face, yaw, pitch, roll, scale, focal_length = 50)

    
    def display_window(self, frame):
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.putText(frame, str(time.time()), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255 ,255 ,255 ), 2, cv2.LINE_AA)
        cv2.imshow('Face Detector ', frame)
        cv2.waitKey(1)

    def run(self, args):
        self.feed.load_data()
        for frame in self.feed.next_batch():
            self.process(frame)
            log.info(msg= 'Frame processing batch image')

        self.feed.close
        
        # Release resources
        cv2.destroyAllWindows()

def main():
    args = build_argparser().parse_args()

    log.basicConfig(format="[ %(levelname)s ] %(asctime) - 15s %(message)s ",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

    log.debug(str(args))

    visualize = Visualize(args)
    visualize.run(args)


if __name__ == '__main__':
    main()
