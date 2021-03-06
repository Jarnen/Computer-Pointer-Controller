
from openvino.inference_engine import IENetwork, IECore
import cv2
import logging as log
import numpy as np
from model_module import Model

class FaceDetection(Model):
    """Class for the Face Detection Model."""

    def __init__(self, model_name, device, threshold):
        """ Initialise and loads the face detecion model"""

        super(FaceDetection, self).__init__(model_name, device, threshold)

    def predict(self, image):
        """Performs inference on image and returns list of faces."""

        width = image.shape[1]
        height = image.shape[0]
        
        image = self.preprocess_input(image)
        assert len(image.shape) == 4, "Image shape should be [1, c, h, w]"
        assert image.shape[0] == 1
        assert image.shape[1] == 3

        input_dict={self.input_blob:image}
        
        #------async inference ---
        infer_request_handle = self.net.start_async(request_id=0, inputs=input_dict)
        if self.net.requests[0].wait(-1) == 0:
            res = infer_request_handle.outputs[self.output_blob]
        
        # -----sync inference ----
        # res = self.net.infer(input_dict)
        # res = res[self.output_blob]
        
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

    def preprocess_output(self, image, rois):
        """
        Crops first face from image and returns it.
        
        Args:
        image   -- the ndarray image.
        rois    -- the list of faces detected from inference.
        
        Returns:
        cropped_roi -- cropped face region of interest.
        """
        # assert len(rois) != 0, "No face detected in the frame."
        log.info(msg= '{} face/s detected..'.format(len(rois)))

        #roi = im[y1:y2, x1:x2]
        cropped_roi = image[rois[0][1]: rois[0][3], rois[0][0]:rois[0][2]]

        return cropped_roi
    