from argparse import ArgumentParser
from utils import draw_axes
import logging as log
import sys
import cv2
import numpy as np
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from landmarks_detection import LandmarksDetection
from gaze_estimation import GazeEstimation
from utils import enhance_frame, increase_bright_contrast
import time
import os



DEVICE_TYPES = ['CPU', 'GPU', 'FPGA', 'MYRAID', 'HETERO:GPU,CPU']
SPEED = ['fast', 'medium', 'slow']
PRECISION = ['high', 'medium', 'low']

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument('-i', '--input_type', metavar='PATH', default=None,
                                  help="(Optional) Specify the type of input " \
                                      "('cam' for camera, default)")
    parser.add_argument('-f', '--input_file', metavar='PATH', default=None,
                                   help="(optional) Path to the input file " \
                                       "(None for input file, default)")
    parser.add_argument('-m_fd', metavar="PATH", default="", required=True,
                        help="Path to the Face Detection model XML file")
    parser.add_argument('-m_lm', metavar="PATH", default="", required=True,
                        help="Path to the Landmarks Detection model XML file")
    parser.add_argument('-m_hp', metavar="PATH", default="", required=True,
                        help="Path to the Pose Estimation model XML file")
    parser.add_argument('-m_ge', metavar="PATH", default="", required=True,
                        help="Path to the Gaze Estimation model XML file")
    
    parser.add_argument('-d_fd', '--device_fd', default='CPU', choices=DEVICE_TYPES,
                       help="(optional) Target device for the " \
                       "Face Detection model device (default: %(default)s)")
    
    parser.add_argument('-d_lm', '--device_lm', default='CPU', choices=DEVICE_TYPES,
                       help="(optional) Target device for the " \
                       "Landmarks Detection model device (default: %(default)s)")

    parser.add_argument('-d_hp', '--device_hp', default='CPU', choices=DEVICE_TYPES,
                       help="(optional) Target device for the " \
                       "Head pose detection model device (default: %(default)s)")

    parser.add_argument('-d_ge', '--device_ge', default='CPU', choices=DEVICE_TYPES,
                       help="(optional) Target device for the " \
                       "Gace estimation model device (default: %(default)s)")

    parser.add_argument('-t','--threshold', metavar='[0..1]', type=float, default=0.5,
                       help="(optional) Probability threshold for face detections" \
                       "(default: %(default)s)")
    
    parser.add_argument('-bs','--batch_size', metavar='[1..10]', type=int, default=2,
                       help="(optional) Batch size for input feeder." \
                       "(default: %(default)s)")

    parser.add_argument('-s','--speed', default='fast', choices=SPEED,
                       help="(required) Speed ('fast', 'medium','slow')for mouse movement" \
                       "(default: %(default)s)")
    parser.add_argument('-p','--precision', default='high', choices=PRECISION,
                       help="(required) Speed ('high', 'medium','low')for mouse movement" \
                       "(default: %(default)s)")
    parser.add_argument('--no_show', action='store_true',
                         help="(optional) Do not display output")
    parser.add_argument('--output_path', default='results/',
                        help="Path to the stats results file")
    parser.add_argument('-v', '--verbose', action='store_true',
                       help="(optional) Be more verbose")
    
    return parser

class Visualize:

    def __init__(self, args):
        """ Initialise the class variables"""
        
        model_initialisation_start = time.time()
        self.face_detector = FaceDetection(args.m_fd, args.device_fd, args.threshold)
        self.landmarks_detector = LandmarksDetection(args.m_lm, args.device_lm, args.threshold)
        self.head_pose_estimation = HeadPoseEstimation(args.m_hp, args.device_hp, args.threshold)
        self.gaze_estimation = GazeEstimation(args.m_ge,args.device_ge, args.threshold)
        self.model_initialised_time = time.time() - model_initialisation_start
        log.info("All model classes initialised in {} milliseconds".format(int(round(self.model_initialised_time * 1000))))

        self.mouse_controller = MouseController(args.precision, args.speed)
        log.info("Mouse controller initialised")

        models_load_start = time.time()
        self.face_detector.load_model()
        self.landmarks_detector.load_model()
        self.head_pose_estimation.load_model()
        self.gaze_estimation.load_model()
        self.models_load_time = time.time() - models_load_start
        log.info("All models loaded sucessfully in {} milliseconds.".format(int(round(self.models_load_time) * 1000)))

        self.feed = InputFeeder(args.input_type, args.input_file, args.batch_size)
        log.info("Input feeder initialised")

        self.show_results = not args.no_show
        self.output_path = args.output_path

        self.total_processing_time = 0
        self.frames_counter = 0
        
    def process_pipeline(self, frame):
        """ Performs pipeline inference on the frame and returns x,y coordinates of gaze direction. """

        assert len(frame.shape) == 3, "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], "Expected BGR or BGRA input"
        
        pipeline_process_start = time.time()
        self.rois = self.face_detector.predict(frame)
        self.face = self.face_detector.preprocess_output(frame, self.rois)
        
        self.landmarks = self.landmarks_detector.predict(self.face)
        self.right_eye_image, self.left_eye_image, self.right_eye_roi, self.left_eye_roi = self.landmarks_detector.preprocess_output(self.face, self.landmarks)
        self.head_pose_angles = self.head_pose_estimation.predict(self.face)
        
        x, y = self.gaze_estimation.predict(self.right_eye_image, self.left_eye_image, self.head_pose_angles)
        
        self.pipeline_processtime = time.time() - pipeline_process_start
        self.total_processing_time += self.pipeline_processtime
        self.frames_counter += 1
        log.info("Completed frame pipleline process in {} seconds. ".format(self.pipeline_processtime))

        return x, y       

    def draw_face_roi(self, frame, roi):
        """
        Draw the face ROI border
        """
        frame = cv2.rectangle(frame, (roi[0][0], roi[0][1]), (roi[0][2], roi[0][3]), (0,220,0) , 2)
    
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

    def display_results(self, frame):
        if self.show_results:
            self.draw_pose_detection(self.face, self.head_pose_angles)
            self.draw_eye_landmarks(self.face, self.right_eye_roi, self.left_eye_roi)
            self.draw_face_roi(frame, self.rois)

        total_processing_pipeline = "Pipeline processed frame in {} milliseconds".format(int(round(self.pipeline_processtime *1000)))
        model_init_time = "All models classes initialised in {} milliseconds".format(int(round(self.model_initialised_time * 1000)))
        model_load_time = "All models loaded in {} milliseconds".format(int(round(self.models_load_time * 1000)))

        cv2.putText(frame, model_init_time , (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,220,0), 1)
        cv2.putText(frame, model_load_time , (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,220,0), 1)
        cv2.putText(frame, total_processing_pipeline , (15, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,220,0), 1)
    
        cv2.imshow('Computer Pointer Controller', frame)
        cv2.waitKey(1)
    
    def run(self, args):
        self.feed.load_data()
        for frame in self.feed.next_batch():
            #enhanced_frame  = enhance_frame(frame) / increase_bright_contrast(frame)
            log.info(msg= 'Frame processing starts')
            x, y = self.process_pipeline(frame)
            
            self.mouse_controller.move(x,y)
            self.display_results(frame)

            fps = self.frames_counter/self.total_processing_time
            with open(os.path.join(self.output_path, 'stats.txt'), 'w') as f:
                f.write('Total pipeline inference time '+ str(self.total_processing_time)+'\n')
                f.write('Frames processed per seconds ' + str(fps)+'\n')
                f.write('Total models load time ' + str(self.models_load_time)+'\n')

            #---stop timer for benchmark async and sync infer---
            # if self.total_processing_time >= 25.0:
            #     break
            # if self.frames_counter >= 25:
            #     break

        self.feed.close
        # Release resources
        log.info(msg= 'Frame processing ends')
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
