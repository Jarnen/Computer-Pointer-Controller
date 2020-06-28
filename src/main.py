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

DEVICE_TYPES = ['CPU', 'GPU', 'FPGA', 'MYRAID', 'HETERO']

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument('-i', '--input_type', metavar='PATH', default='cam',
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
    parser.add_argument('-t','--threshold', metavar='[0..1]', type=float, default=0.6,
                       help="(optional) Probability threshold for face detections" \
                       "(default: %(default)s)")
    parser.add_argument('-v', '--verbose', action='store_true',
                       help="(optional) Be more verbose")
    
    return parser

class ProcessFrame:

    def __init__(self, args):
        self.face_detector = FaceDetection(args.m_fd, args.device, args.lib, args.threshold)
        self.landmarks_detector = LandmarksDetection(args.m_ld, args.device, args.lib, args.threshold)
        self.head_pose_estimation = HeadPoseEstimation(args.m_pe, args.device, args.threshold)
        self.gaze_estimation = GazeEstimation(args.m_ge,args.device, args.threshold)
        self.mouse_controller = MouseController('high', 'fast')

        self.face_detector.load_model()
        self.landmarks_detector.load_model()
        self.head_pose_estimation.load_model()
        self.gaze_estimation.load_model()

    def process(self, frame):
        assert len(frame.shape) == 3, "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], "Expected BGR or BGRA input"

        rois = self.face_detector.predict(frame)
        face = self.face_detector.preprocess_output(frame, rois)
        
        landmarks = self.landmarks_detector.predict(face)
        right_eye_image, left_eye_image = self.landmarks_detector.preprocess_output(face, landmarks)

        head_pose_angles = self.head_pose_estimation.predict(face)
        
        x, y = self.gaze_estimation.predict(left_eye_image, right_eye_image, head_pose_angles)
        #gaze_result = self.gaze_estimation.predict(gaze_estimation_inputs)
        # x, y = self.gaze_estimation.predict(gaze_estimation_results)

        self.mouse_controller.move(x,y)


class Visualize:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord('q'), ord('Q'), 27}

    def __init__(self, args):
        """
        Initialise the variables
        """
        self.process_frame = ProcessFrame(args)

    def frame_detector(self, frame):
        """
        Detects and returns face coordinates on the frame
        """


    def draw_face_roi(self, frame, roi):
        """
        Draw the face ROI border
        """
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0,220,0) , 2)
    
    def draw_landmarks_detection(self, frame, crop_right_eye, crop_left_eye):
        """
        Draw the eyes ROI landmarks
        """
    
    def draw_pose_detection(self, frame, center, yaw, pitch, roll, focal_length):
        """
        Draw yaw, pitch, roll angle lines for visualisation
        """
        scale = 2
        draw_axes(frame, center, yaw, pitch, roll, scale, focal_length = 50)
    
    def display_window(self, frame):
        color = (255 ,255 ,255 )
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        thickness = 2
        text = "Press '%s' key to exit " % (self.BREAK_KEY_LABELS)
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])        
        cv2.putText(frame, text, tuple(origin.astype(int)), font, text_scale, color, thickness)
        
    def run(self, args):
        feed = InputFeeder(args.input_type, args.input_file)
        feed.load_data()
        for batch in feed.next_batch():
            self.process_frame.process(batch)
        feed.close
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
