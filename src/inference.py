
from openvino.inference_engine import IENetwork, IECore
import cv2
import logging as log

class Model:
    '''
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    '''
    def __init__(self, model_name, device, extension):
        '''
        Set all the instance variables
        '''
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extension
        self.net = None
        self.plugins = None

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
        Load the model to the device specified by the user and add the extension
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
        raise NotImplementedError

    def check_model(self, net, device):
        """
        Check model if correct extensions added
        """
        plugin = self.plugins[device]

        if plugin.device == "CPU":
            supported_layers = plugin.get_supported_layers(net)
            not_supported_layers = [l for l in net.layers.keys() \
                                    if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("The following layers are not supported " \
                    "by the plugin for the specified device {}:\n {}" \
                    .format(plugin.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions " \
                    "library path in the command line parameters using " \
                    "the '-l' parameter")
                raise NotImplementedError(
                    "Some layers are not supported on the device")

    def preprocess_input(self, image):
        """
        Set and returns input image to CHW format for inferencing 
        """
        pr_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        pr_image = pr_image.transpose((2,0,1)) #transpose layout from HWC to CHW
        pr_image = pr_image.reshape(1, *pr_image.shape)

        return pr_image

    def preprocess_output(self, outputs):
        raise NotImplementedError
