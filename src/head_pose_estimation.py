
from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
from model_module import Model

class HeadPoseEstimation(Model):
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device, threshold):
        """ Initialise and loads the head pose detecion model"""
        super(HeadPoseEstimation, self).__init__(model_name, device, threshold)


    def predict(self, image):
        """
        Performs inference on face image and returns a preprocessed list of output angles.
        
        Args:
        image   --- cropped face image

        Returns;
        dictionary   --- [[yaw,pitch,roll]]
        """

        image = self.preprocess_input(image)
        assert len(image.shape) == 4, "Image shape should be [1, c, h, w]"
        assert image.shape[0] == 1
        assert image.shape[1] == 3

        input_dict={self.input_blob:image}
        #----async inference ----
        infer_request_handle = self.net.start_async(request_id=0, inputs=input_dict)
        if self.net.requests[0].wait(-1) == 0:
            res = infer_request_handle.outputs
            output = self.preprocess_output(res)

        #----sync inference ----
        # res = self.net.infer(input_dict)
        # output = self.preprocess_output(res)
        
        return output

    def preprocess_output(self, outputs):
        """
        Preprocess outputs

        Args:
        outputs --- dictionary of pose angles.

        Returns:
        dictionary   --- [[yaw,pitch,roll]]
        """
        angle_y_fc = outputs.get('angle_y_fc')
        angle_p_fc = outputs.get('angle_p_fc')
        angle_r_fc = outputs.get('angle_r_fc')

        yaw = angle_y_fc[0][0]
        pitch = angle_p_fc[0][0]
        roll = angle_r_fc[0][0]

        output = np.array([[yaw, pitch, roll]])

        return output
        
