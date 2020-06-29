
from openvino.inference_engine import IENetwork, IECore 
import cv2
import logging as log
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''


class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device, extensions):
        """
        Set the instance variables
        """
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extensions
        self.net = None

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not initialised the network. Have you enterred the correct model path?")

        assert len(self.model.inputs) == 3, "Expected 3 input blob"
        assert len(self.model.outputs) == 1, "Expected 1 output blob"

        self.input_blob = next(iter(self.model.inputs)) 
        self.output_blob = next(iter(self.model.outputs))
        self.input_shape = self.model.inputs[self.input_blob].shape
        self.output_shape = self.model.outputs[self.output_blob].shape

    def load_model(self):
        """
        Loads the model 
        """
        core = IECore()
        self.net = core.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        """
        Runs inference and returns the raw results

        Args:
        inputs: input dictionary in the format {'head_pose_angles': head_pose_angles, 'left_eye_image': left_eye_image, 'right_eye_image': right_eye_image}
        """

        left_eye_image = self.preprocess_image(left_eye_image)
        right_eye_image = self.preprocess_image(right_eye_image)

        inputs = {'head_pose_angles': head_pose_angles, 'left_eye_image': left_eye_image, 'right_eye_image': right_eye_image}

        # infer_request_handle = self.net.start_async(request_id=0, inputs=inputs)
        # #infer_status = infer_request_handle.wait()
        # result = infer_request_handle.outputs
        result  = self.net.infer(inputs)
        x, y = self.preprocess_output(result)
        
        return x, y  

    def check_model(self):
        raise NotImplementedError
     
    '''
    def preprocess_input(self, left_eye_image, right_eye_image, head_pose_angles):
        """
        Returns input dictionary for inferencing

        Args:
        left_eye_image: cropped left eye image in the format [H,W,C].
        right_eye_image: cropped right eye image in the format [H,W,C].
        head_pose_angles: head pose angles in the format [1,3].
        """

        left_eye_image = self.preprocess_image(left_eye_image)
        right_eye_image = self.preprocess_image(right_eye_image)

        inputs = {'head_pose_angles': head_pose_angles, 'left_eye_image': left_eye_image, 'right_eye_image': right_eye_image}

        return inputs
        '''

    def preprocess_image(self, image):
        image_shape = image.shape
        log.info("Image shape: {}".format(image_shape))
        pr_image = cv2.resize(image, (60, 60))
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
