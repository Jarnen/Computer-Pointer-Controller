'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
from model_module import Model

class LandmarksDetection(Model):
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device, threshold):
        super(LandmarksDetection, self).__init__(model_name, device, threshold)

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

        # #----async inference----
        # infer_request_handle = self.net.start_async(request_id=0, inputs=input_dict)
        # if self.net.requests[0].wait(-1) == 0:
        #     result = infer_request_handle.outputs[self.output_blob]

        #----synce inference----
        results = self.net.infer(input_dict)
        result = results[self.output_blob]

        return result

    def preprocess_output(self, image, result):
        '''
        Returns the (x,y) coordinates of right and left eye respectively
        '''
        assert result.size == 10, "Result passed in is not of size 10"
        
        iw, ih = image.shape[:-1]
        
        right_eye = (np.int(ih*result[0][0]), np.int(iw*result[0][1]))
        left_eye = (np.int(ih*result[0][2]), np.int(iw*result[0][3]))

        #scale = 10 #10 worked fine
        scale = 10
        right_eye_xmin = right_eye[0] - scale
        right_eye_ymin = right_eye[1] - scale
        right_eye_xmax = right_eye[0] + scale
        right_eye_ymax = right_eye[1] + scale
        right_eye_roi = [right_eye_xmin, right_eye_ymin,right_eye_xmax, right_eye_ymax]

        left_eye_xmin = left_eye[0] - scale
        left_eye_ymin = left_eye[1] - scale
        left_eye_xmax = left_eye[0] + scale
        left_eye_ymax = left_eye[1] + scale
        left_eye_roi = [left_eye_xmin, left_eye_ymin,left_eye_xmax, left_eye_ymax]

        crop_right_eye = image[right_eye_ymin:right_eye_ymax, right_eye_xmin:right_eye_xmax]
        crop_left_eye = image[left_eye_ymin:left_eye_ymax, left_eye_xmin:left_eye_xmax]
        
        return crop_right_eye, crop_left_eye, right_eye_roi, left_eye_roi
        