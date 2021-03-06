
from openvino.inference_engine import IENetwork, IECore 
import cv2
import logging as log
from model_module import Model

class GazeEstimation(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device, threshold):
        super(GazeEstimation, self).__init__(model_name,device, threshold)

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        """
        Performs inference and returns x, y coordinates

        Args:
        inputs: left eye roi, right eye roi, head pose angles
        """

        left_eye_image = self.preprocess_input(left_eye_image)
        right_eye_image = self.preprocess_input(right_eye_image)

        inputs = {'head_pose_angles': head_pose_angles, 'left_eye_image': left_eye_image, 'right_eye_image': right_eye_image}

        #----async inference----
        infer_request_handle = self.net.start_async(request_id=0, inputs=inputs)
        if self.net.requests[0].wait(-1) == 0:
            result = infer_request_handle.outputs
            x, y = self.preprocess_output(result)

        #----sync inference----
        # results = self.net.infer(inputs)
        # x, y = self.preprocess_output(results)   
        
        return x, y  
 
    def preprocess_input(self, frame):
        """
        Processed eye landmark and returns eye image [1,2,60,60]

        Arg:
        Eye frame extracted from landmarks
        """
        frame_shape = frame.shape
        
        log.info("Image shape: {}".format(frame_shape))
        pr_image = cv2.resize(frame, (60, 60))
        pr_image = pr_image.transpose((2,0,1)) #transpose layout from HWC to CHW
        pr_image = pr_image.reshape(1, *pr_image.shape)

        return pr_image

    def preprocess_output(self, outputs):
        """
        Returns the x and y points from the gaze vector

        Args:
        outputs: inference result from the inference 
        """
        x = outputs.get('gaze_vector')[0][0]
        y = outputs.get('gaze_vector')[0][1]

        return x,y
