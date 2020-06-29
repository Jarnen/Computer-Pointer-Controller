'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np

class FaceDetection():
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device, extensions, threshold):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extensions
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
        #core.add_extension(self.extension, self.device)
        self.net = core.load_network(network=self.model, device_name=self.device, num_requests=1)

        return

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        #width, height = image.shape[:-1]
        width = image.shape[1]
        height = image.shape[0]
        
        image = self.preprocess_input(image)
        assert len(image.shape) == 4, "Image shape should be [1, c, h, w]"
        assert image.shape[0] == 1
        assert image.shape[1] == 3

        input_dict={self.input_blob:image}
        res = self.net.infer(input_dict)
        res = res[self.output_blob]
        output_data = res[0][0]
        rois = []
        for _, proposal in enumerate(output_data):
            if proposal[2] > 0.5:
                xmin = np.int(width * proposal[3])
                ymin = np.int(height * proposal[4])
                xmax = np.int(width * proposal[5])
                ymax = np.int(height * proposal[6])
                rois.append([xmin,ymin,xmax,ymax])
        return rois

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''

        pr_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        pr_image = pr_image.transpose((2,0,1)) #transpose layout from HWC to CHW
        pr_image = pr_image.reshape(1, *pr_image.shape)
    
        return pr_image

    def preprocess_output(self, image, rois):
        '''
        Crops the region of interest (face) from the image and returns it.

        '''
        assert len(rois) != 0, "No face detected in the frame."

        cropped_roi = image[rois[0][0]: rois[0][2], rois[0][1]:rois[0][3]]

        return cropped_roi