# Computer Pointer Controller

*TODO:* Write a short introduction to your project.

This project demonstrates a computer pointer controller by implementing a pipeline of four different Intel's pretrained models. These models are face detection model, landmarks regression model, head pose estimation model and gaze estimation model.

It detects a face from camera or video using the face detection model. From the face, facial landmarks, right eye and left eye, are detected using the landmarks regression model. Also the head pose angles, yaw, pitch and roll, are detected using the head pose estimation model. The landmarks together with head pose angles are then used by gaze estimation model to predict the direction of gaze by the person. This direction is then used by the pyautogui to move the computer pointer to where the person is gazing.

![Sample Image](cpc_image.png)
Format: ![Alt Text](url)


## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

Below is the project structure:
```
📦ComputerPointerController
 ┣ 📂bin
 ┃ ┣ 📜.gitkeep
 ┃ ┗ 📜demo.mp4
 ┣ 📂cpc-project-env
 ┃ ┗ 📜pyvenv.cfg
 ┣ 📂models
 ┃ ┗ 📂intel
 ┃ ┃ ┣ 📂face-detection-adas-0001
 ┃ ┃ ┃ ┣ 📂FP16
 ┃ ┃ ┃ ┃ ┣ 📜face-detection-adas-0001.bin
 ┃ ┃ ┃ ┃ ┗ 📜face-detection-adas-0001.xml
 ┃ ┃ ┃ ┗ 📂FP32
 ┃ ┃ ┃ ┃ ┣ 📜face-detection-adas-0001.bin
 ┃ ┃ ┃ ┃ ┗ 📜face-detection-adas-0001.xml
 ┃ ┃ ┣ 📂face-detection-adas-binary-0001
 ┃ ┃ ┃ ┗ 📂INT1
 ┃ ┃ ┃ ┃ ┣ 📜face-detection-adas-binary-0001.bin
 ┃ ┃ ┃ ┃ ┗ 📜face-detection-adas-binary-0001.xml
 ┃ ┃ ┣ 📂gaze-estimation-adas-0002
 ┃ ┃ ┃ ┣ 📂FP16
 ┃ ┃ ┃ ┃ ┣ 📜gaze-estimation-adas-0002.bin
 ┃ ┃ ┃ ┃ ┗ 📜gaze-estimation-adas-0002.xml
 ┃ ┃ ┃ ┗ 📂FP32
 ┃ ┃ ┃ ┃ ┣ 📜gaze-estimation-adas-0002.bin
 ┃ ┃ ┃ ┃ ┗ 📜gaze-estimation-adas-0002.xml
 ┃ ┃ ┣ 📂head-pose-estimation-adas-0001
 ┃ ┃ ┃ ┣ 📂FP16
 ┃ ┃ ┃ ┃ ┣ 📜head-pose-estimation-adas-0001.bin
 ┃ ┃ ┃ ┃ ┗ 📜head-pose-estimation-adas-0001.xml
 ┃ ┃ ┃ ┗ 📂FP32
 ┃ ┃ ┃ ┃ ┣ 📜head-pose-estimation-adas-0001.bin
 ┃ ┃ ┃ ┃ ┗ 📜head-pose-estimation-adas-0001.xml
 ┃ ┃ ┗ 📂landmarks-regression-retail-0009
 ┃ ┃ ┃ ┣ 📂FP16
 ┃ ┃ ┃ ┃ ┣ 📜landmarks-regression-retail-0009.bin
 ┃ ┃ ┃ ┃ ┗ 📜landmarks-regression-retail-0009.xml
 ┃ ┃ ┃ ┗ 📂FP32
 ┃ ┃ ┃ ┃ ┣ 📜landmarks-regression-retail-0009.bin
 ┃ ┃ ┃ ┃ ┗ 📜landmarks-regression-retail-0009.xml
 ┣ 📂src
 ┃ ┣ 📜face_detection.py
 ┃ ┣ 📜gaze_estimation.py
 ┃ ┣ 📜head_pose_estimation.py
 ┃ ┣ 📜input_feeder.py
 ┃ ┣ 📜landmarks_detection.py
 ┃ ┣ 📜main.py
 ┃ ┣ 📜model.py
 ┃ ┣ 📜model_module.py
 ┃ ┣ 📜mouse_controller.py
 ┃ ┗ 📜utils.py
 ┣ 📜README.md
 ┣ 📜cpc_image.png
 ┗ 📜requirements.txt
```
## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
