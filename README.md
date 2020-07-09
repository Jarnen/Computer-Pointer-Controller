# Computer Pointer Controller

*TODO:* Write a short introduction to your project.

This project demonstrates a computer pointer controller by implementing a pipeline of four different Intel's pretrained models. These models are face detection model, landmarks regression model, head pose estimation model and gaze estimation model. Refer below the list of the models;

        1. face-detection-adas-binary-0001
        2. gaze-estimation-adas-0002
        3. head-pose-estimation-adas-0001
        4. landmarks-regression-retail-0009

This project implements a pipeline of models input and outputs. It takes a video or camera feed as an input for the face detection model and outputs the face region of interests (roi). Landmarks regression model gets the face roi as an input and detects facial landmarks, right eye and left eye. Also the head pose estimation model takes the facial roi as input and detects the head pose angles, yaw, pitch and roll. The landmarks and the head pose angles are then fed to the gaze estimation model to predict the direction or cordinates of gaze by the person. These coordinates are then fed to the pyautogui to move the computer pointer to where the person is gazing.

![Sample Image](cpc_image.png)
Format: ![Alt Text](url)


## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

This project comes together with it's virtual environment. Refer to the project structure below, the virtual environment is the cpc-project-env folder. Once you have downloaded this repository, change directory into it and run the command "source cpc-project-env/bin/activate" to activate the environment. 

However, to run on your own environment, refer to the requirements.txt file for all the required software that are needed to be installed before executing. This project had been tested and executed on a Linux operating system with Intel's Openvino Toolkit 2020.1. The hardware used had an Intel Core i7 processor (8th Gen) with integrated GPU. 

Refer below is the project structure:

The bin folder contains the demo video file, the models folder contains all the Intel's Pretrained models needed for execution, and the src folder contains all necessary python files.
```
📦ComputerPointerController
 ┣ 📂bin
 ┃ ┗ 📜demo.mp4
 ┣ 📂cpc-project-env
 ┃ ┣ 📂bin
 ┃   ┣ 📜activate
 ┃  
 ┣ 📂models
 ┃ ┗ 📂intel
 ┃   ┣ 📂face-detection-adas-0001
 ┃   ┃ ┣ 📂FP16
 ┃   ┃ ┃ ┣ 📜face-detection-adas-0001.bin
 ┃   ┃ ┃ ┗ 📜face-detection-adas-0001.xml
 ┃   ┃ ┗ 📂FP32
 ┃   ┃ ┃ ┣ 📜face-detection-adas-0001.bin
 ┃   ┃ ┃ ┗ 📜face-detection-adas-0001.xml
 ┃   ┣ 📂face-detection-adas-binary-0001
 ┃   ┃ ┗ 📂INT1
 ┃   ┃ ┃ ┣ 📜face-detection-adas-binary-0001.bin
 ┃   ┃ ┃ ┗ 📜face-detection-adas-binary-0001.xml
 ┃   ┣ 📂gaze-estimation-adas-0002
 ┃   ┃ ┣ 📂FP16
 ┃   ┃ ┃ ┣ 📜gaze-estimation-adas-0002.bin
 ┃   ┃ ┃ ┗ 📜gaze-estimation-adas-0002.xml
 ┃   ┃ ┗ 📂FP32
 ┃   ┃ ┃ ┣ 📜gaze-estimation-adas-0002.bin
 ┃   ┃ ┃ ┗ 📜gaze-estimation-adas-0002.xml
 ┃   ┣ 📂head-pose-estimation-adas-0001
 ┃   ┃ ┣ 📂FP16
 ┃   ┃ ┃ ┣ 📜head-pose-estimation-adas-0001.bin
 ┃   ┃ ┃ ┗ 📜head-pose-estimation-adas-0001.xml
 ┃   ┃ ┗ 📂FP32
 ┃   ┃ ┃ ┣ 📜head-pose-estimation-adas-0001.bin
 ┃   ┃ ┃ ┗ 📜head-pose-estimation-adas-0001.xml
 ┃   ┗ 📂landmarks-regression-retail-0009
 ┃     ┣ 📂FP16
 ┃     ┃ ┣ 📜landmarks-regression-retail-0009.bin
 ┃     ┃ ┗ 📜landmarks-regression-retail-0009.xml
 ┃     ┗ 📂FP32
 ┃       ┣ 📜landmarks-regression-retail-0009.bin
 ┃       ┗ 📜landmarks-regression-retail-0009.xml
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

Use the command line tool in Windows or terminal in Linux to execute the application. To run the application, change directory into the root folder and execute the following commands:

    1. Sample 1: Using CPU to run inference on the frames
```
python3 src/main.py -i 'cam'  -m_fd "models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001" -m_lm "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009" -m_hp "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001" -m_ge "models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002" --device_fd 'CPU' --device_lm 'CPU' --device_ge 'CPU' --device_hp 'CPU'  --verbose
```
    2. Sample 2: Using GPU to run inference on the frames
```
python3 src/main.py -i 'cam'  -m_fd "models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001" -m_lm "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009" -m_hp "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001" -m_ge "models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002" --device_fd 'GPU' --device_lm 'GPU' --device_ge 'GPU' --device_hp 'GPU'  --verbose
```
For help on which arguments to use, execute the below command;
```
        python3 src/main.py --help
```

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.


## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

Models load time and inference time were performed on two different hardware. They are;
    
    1. Intergrated GPU - GeForce MX150/PCIe/SSE2, and 
    2. CPU - Intel® Core™ i7-8550U CPU @ 1.80GHz × 8. 

Below are different scenarios.

    1. Scenario 1: CPU Hardware used for all Models 
    
Sample command executed;

```
python3 src/main.py -i 'cam'  -m_fd "models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001" -m_lm "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009" -m_hp "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001" -m_ge "models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002" --device_fd 'CPU' --device_lm 'CPU' --device_ge 'CPU' --device_hp 'CPU'  --verbose --output_path 'results'
```
Refer to the results table in the Results section for detailed results.


    2. Scenario 2: GPU Hardware used for all Models
    
Sample command executed;

```
python3 src/main.py -i 'cam'  -m_fd "models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001" -m_lm "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009" -m_hp "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001" -m_ge "models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002" --device_fd 'GPU' --device_lm 'GPU' --device_ge 'GPU' --device_hp 'GPU'  --verbose --output_path 'results'
```
Refer to the results table in the Results section for detailed results.

Another benchmark tests were done on multiple precisions. Below are the different scenarios.

    1. Scenario 1: Presicion FP32 and CPU Hardware used for all Models
    
Sample command executed;
```
python3 src/main.py -i 'cam'  -m_fd "models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001" -m_lm "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009" -m_hp "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001" -m_ge "models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002" --device_fd 'CPU' --device_lm 'CPU' --device_ge 'CPU' --device_hp 'CPU'  --verbose --output_path 'results'
```

Refer to results table in the Results table for detailed results.

    2. Scenario 2: Precision FP16 and CPU Hardware used for all Models

Sample command executed;

```
python3 src/main.py -i 'cam'  -m_fd "models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001" -m_lm "models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009" -m_hp "models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001" -m_ge "models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002" --device_fd 'CPU' --device_lm 'CPU' --device_ge 'CPU' --device_hp 'CPU'  --verbose --output_path 'results'
```

Refer to results table in the Results table for detailed results.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

### Results Table
Below is the table showing results of the different scenarios performed in pipeline processing of the frames. (Refer lines 200/201 of main.py for breakpoint)

| Hardware | Models Precisions | Total Models Load Time (s) | Total Pipeline Inference Time (s) | Frames Per Second  | Batch Size |
|----------|-------------------|----------------------------|-----------------------------------|--------------------|------------|
| GPU      | FP32              | 42.085306882858276         | 1.3761012554168701                | 18.167267780325226 | 2          |
| GPU      | FP16              | 43.26167893409729          | 0.9960892200469971                | 25.09815335499812  | 1          |
| CPU      | FP16              | 0.366560697555542          | 0.7050163745880127                | 35.460169296931774 | 1          |
| CPU      | FP32              | 0.328019380569458          | 0.760749101638794                 | 32.86234574072501  | 1          |
| CPU      | FP32              | 0.324234962463378          | 0.7439463138580322                | 33.60457540323369  | 1          |
| GPU      | FP32              | 42.11428737640381          | 1.3050618171691895                | 19.156180704319063 | 1          |
| CPU      | FP16              | 0.41742587089538574        | 0.7670192718505859                | 32.59370516164809  | 1          |
| CPU      | FP16              | 0.3760707378387451         | 0.7119126319885254                | 35.1166686425687   | 2          |

From the above results, we concluded as per below;
    1. CPU loads models faster than GPU.
    2. Higher precisions (FP32) takes longer processing time than lower precisions (FP16).

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

Here is the sample command executed to get the results.
```
python3 src/main.py -i 'cam'  -m_fd "models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001" -m_lm "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009" -m_hp "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001" -m_ge "models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002" --device_fd 'CPU' --device_lm 'CPU' --device_ge 'CPU' --device_hp 'CPU'  --verbose --batch_size 2
```
Below are the results of the different inference types and batch sizes used for frames breakpoint at 25 frames. (Refer lines 202/203 of main.py)

| Hardware | Inference Type | Total Models Load Time (s) | Total Pipeline Inference Time (s) | Frames Per Second  | Batch Size |
|----------|----------------|----------------------------|-----------------------------------|--------------------|------------|
| CPU      | synchronous    | 0.3242354393005371         | 25.002384185791016                | 24.037707585565038 | 2          |
| CPU      | asynchronous   | 0.33762550354003906        | 25.023040056228638                | 23.65819655284617  | 2          |
| CPU      | synchronous    | 0.3262178897857666         | 25.010347604751587                | 39.02384786585592  | 1          |
| CPU      | asynchronous   | 0.33506035804748535        | 25.02401304244995                 | 37.56391904069957  | 1          |
| GPU      | asynchronous   | 42.54989671707153          | 25.017953634262085                | 17.66731230146182  | 2          |
| GPU      | synchronous    | 41.91681599617004          | 25.028812408447266                | 17.659647321133953 | 2          |
| GPU      | synchronous    | 42.34746384620666          | 25.03768563270569                 | 17.57350924740609  | 1          |
| GPU      | asynchronous   | 42.30827450752258          | 25.046454906463623                | 17.806911264112784 | 1          |

From the results above, it can be concluded that;
    1. When using CPU hardware, synchronous inference process more frames per second than asychronous inference.
    2. When using GPU hardware, asynchronous inference process more frames per second than synchronous infernce. 

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

#### Multiple People Case
When multiple people are detected, the processing pipeline breaks because other models like landmarks require input from only one face. To overcome this edge case scenario, I have allowed for multiple faces to be detected but only the first face detected selected for further inferencing. The first face is detected and allowed through the pipeline for further inference of landmarks, head pose angles estimation and gaze estimation. All other faces detected are ignored. Refer below the snippet of code from face_detection.py that does that;
```
cropped_roi = image[rois[0][1]: rois[0][3], rois[0][0]:rois[0][2]]
```

#### Lighting Changes Case
In low lighting situations, face cannot be detected and hence causes the inference pipeline to break. I tried to overcome this by applying brightness and contrast to the frame captured. Also another option I used was to convert image to LAB Color model and splitting the LAB image to different channels, applying CLAHE to L-channel, merged the CLAHE enhanced L-channel. Refer to the utils.py file lines 126 to 150 for these two options. I then apply either one of these in the main.py file at line 186. 

Another option was to install a third party software like Gucview (on Linux) to enhance the camera to detect persons in low lighting scenarios. I have also done some research on other options like Dual Illumination Estimation for Robust Exposure Correction and Low-light Image Enhancement via Illumination Map Estimation and will apply later to this project. Refer to links in references for these.

