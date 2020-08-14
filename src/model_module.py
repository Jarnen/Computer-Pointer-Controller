from openvino.inference_engine import IENetwork, IECore
import cv2
import logging as log
import numpy as np


class Model(object):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device, threshold):
        """ Initialise instance variables"""
        
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.net = None

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not initialised the network. Have you enterred the correct model path?")

        self.input_blob = next(iter(self.model.inputs)) 
        self.output_blob = next(iter(self.model.outputs))
        self.input_shape = self.model.inputs[self.input_blob].shape
        self.output_shape = self.model.outputs[self.output_blob].shape

        log.info('Model {} params initialised successfully'.format(self.model.name))
    
    def load_model(self):
        """Loads the device model"""

        core = IECore()
        self.net = core.load_network(network=self.model, device_name=self.device, num_requests=1)

        log.info('Network for model {} loaded suceessfully.'.format(self.model.name))


    def preprocess_input(self, image):
        """Preprocess the input image"""

        log.info("Frame preprocessed successfully.")
        pr_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        pr_image = pr_image.transpose((2,0,1)) #transpose layout from HWC to CHW
        pr_image = pr_image.reshape(1, *pr_image.shape)

        return pr_image
