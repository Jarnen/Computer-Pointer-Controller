source  ~/Projects/ComputerPointerController/cpc-project-env/bin/activate
source /opt/intel/openvino/bin/setupvars.sh
python3 src/main.py -i 'cam'  -m_fd "models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001" -m_lm "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009" -m_hp "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001" -m_ge "models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002" --device_fd 'CPU' --device_lm 'CPU' --device_ge 'CPU' --device_hp 'CPU'  --verbose


