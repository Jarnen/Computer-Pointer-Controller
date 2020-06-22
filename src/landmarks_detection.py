'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np

class LandmarksDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extension
        self.net = None

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not initialised the network. Have you enterred the correct model path?")

        assert len(self.model.inputs) == 1, "Expected 1 input blob"
        assert len(self.model.outputs) == 1, "Expected 1 output blob"

        self.input_blob = next(iter(self.model.inputs)) 
        self.output_blob = next(iter(self.model.outputs))
        self.input_shape = self.model.inputs[self.input_blob].shape
        self.output_shape = self.model.outputs[self.output_blob].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        core = IECore()
        core.add_extension(self.extension)
        self.net = core.load_network(network=self.model, device_name=self.device)

        return

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        assert len(image.shape) == 4, "Image shape should be [1, c, h, w]"
        assert image.shape[0] == 1
        assert image.shape[1] == 3

        input_dict={self.input_name:image}
        result = self.net.infer(input_dict)
        result = result[self.output_name]

        return result

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''

        pr_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        pr_image = pr_image.transpose((2,0,1)) #transpose layout from HWC to CHW
        pr_image = pr_image.reshape(1, *pr_image.shape)
        
        return pr_image

    def preprocess_output(self, image, result):
        '''
        Returns the (x,y) coordinates of right and left eye respectively
        '''
        assert result.size == 10, "Result set is not valid"

        iw, ih = image.shape[:-1]

        right_eye = (np.int(ih*result[0][0]), np.int(iw*result[0][1]))
        left_eye = (np.int(ih*result[0][2]), np.int(iw*result[0][3]))
        
        return right_eye, left_eye