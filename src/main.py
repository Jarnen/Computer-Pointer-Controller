from argparse import ArgumentParser
from utils import draw_axes
import logging as log
import sys
import cv2
import numpy as np

DEVICE_TYPES = ['CPU', 'GPU', 'FPGA', 'MYRAID', 'HETERO']

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()

    general = parser.add_argument_group('General') 
    general.add_argument('-i', '--input', metavar='PATH', default='0',
                                  help="(optional) Path to the input video " \
                                      "('0' for camera, default)")
    general.add_argument('-o', '--output', metavar="PATH", default="",
                         help="(optional) Path to save the output video to")
    
    models = parser.add_argument_group('Models')
    models.add_argument('-m_fd', metavar="PATH", default="", required=True,
                        help="Path to the Face Detection model XML file")
    models.add_argument('-m_ld', metavar="PATH", default="", required=True,
                        help="Path to the Landmarks Detection model XML file")
    models.add_argument('-m_pe', metavar="PATH", default="", required=True,
                        help="Path to the Pose Estimation model XML file")
    models.add_argument('-m_ge', metavar="PATH", default="", required=True,
                        help="Path to the Gaze Estimation model XML file")
    
    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_fd', default='CPU', choices=DEVICE_TYPES,
                       help="(optional) Target device for the " \
                       "Face Detection model (default: %(default)s)")
    infer.add_argument('-d_ld', default='CPU', choices=DEVICE_TYPES,
                       help="(optional) Target device for the " \
                       "Landmarks Detection model (default: %(default)s)")
    infer.add_argument('-d_pe', default='CPU', choices=DEVICE_TYPES,
                       help="(optional) Target device for the " \
                       "Pose Estimation model (default: %(default)s)")
    infer.add_argument('-d_ge', default='CPU', choices=DEVICE_TYPES,
                       help="(optional) Target device for the " \
                       "Gaze Estimation model (default: %(default)s)")
    infer.add_argument('-l', '--cpu_lib', metavar="PATH", default="",
                       help="(optional) For MKLDNN (CPU)-targeted custom layers, if any. " \
                       "Path to a shared library with custom layers implementations")
    infer.add_argument('-c', '--gpu_lib', metavar="PATH", default="",
                       help="(optional) For clDNN (GPU)-targeted custom layers, if any. " \
                       "Path to the XML file with descriptions of the kernels")
    infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
                       help="(optional) Probability threshold for face detections" \
                       "(default: %(default)s)")
    infer.add_argument('-t_ld', metavar='[0..1]', type=float, default=0.6, 
                       help="(optional) Probability threshold for landmarks detections" \
                       "(default: %(default)s)")
    infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
                       help="(optional) Scaling ratio for bboxes passed to face recognition " \
                       "(default: %(default)s)")
    infer.add_argument('-v', '--verbose', action='store_true',
                       help="(optional) Be more verbose")
    
    return parser

class ProcessFrame:

    def __init__(self, args):
        devices = set([args.d])
class Visualize:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord('q'), ord('Q'), 27}

    def __init__(self):
        """
        Initialise the variables
        """
    
    def frame_detector(frame):
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
        draw_axes(frame, center, yaw, pitch, roll, focal_length = 50)
    
    def display_window(self, frame):
        color = (255 ,255 ,255 )
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        thickness = 2
        text = "Press '%s' key to exit " % (self.BREAK_KEY_LABELS)
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])        
        cv2.putText(frame, text, tuple(origin.astype(int)), font, text_scale, color, thickness)

    def process(self, input_stream):
        self.input_stream = input_stream

        while input_stream.isOpened():
            ret_val, frame = input_stream.read()
            if not ret_val:
                break

            face_detection = self.frame_detector(frame)


    
    @staticmethod
    def open_input_stream(path):
        log.info("Reading input data from '%s'" % (path))
        stream = path
        try:
            stream = int(path)
        except ValueError:
            pass
        return cv2.VideoCapture(stream)
        
    def run(self, args):
        input_stream = Visualize.open_input_stream(args.input)
        if input_stream is None or not input_stream.isOpened():
            log.error("Cannot open input stream: %s" % args.input)
        fps = input_stream.get(cv2.CAP_PROP_FPS)
        frame_size = (int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        log.info("Input stream info: %d x %d @ %.2f FPS" % \
            (frame_size[0], frame_size[1], fps))
        
        self.process(input_stream)

        # Release resources
        if input_stream:
            input_stream.release()

        cv2.destroyAllWindows()

def main():
    args = build_argparser().parse_args

    log.basicConfig(format="[ %(levelname)s ] %(asctime) - 15s %(message)s ",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

    log.debug(str(args))

    visualize = Visualize(args)
    visualize.run(args)


if __name__ == '__main__':
    main()
