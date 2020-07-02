'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
from model_module import Model

class HeadPoseEstimation(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device, threshold):
        super(HeadPoseEstimation, self).__init__(model_name, device, threshold)


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        image = self.preprocess_input(image)
        assert len(image.shape) == 4, "Image shape should be [1, c, h, w]"
        assert image.shape[0] == 1
        assert image.shape[1] == 3

        input_dict={self.input_blob:image}
        #results = self.net.infer(input_dict)
        infer_request_handle = self.net.start_async(request_id=0, inputs=input_dict)
        if self.net.requests[0].wait(-1) == 0:
            res = infer_request_handle.outputs
            output = self.preprocess_output(res)
        
        return output

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        angle_y_fc = outputs.get('angle_y_fc')
        angle_p_fc = outputs.get('angle_p_fc')
        angle_r_fc = outputs.get('angle_r_fc')

        yaw = angle_y_fc[0][0]
        pitch = angle_p_fc[0][0]
        roll = angle_r_fc[0][0]

        output = np.array([[yaw, pitch, roll]])

        return output
        
