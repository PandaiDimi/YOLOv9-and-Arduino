# YOLOv9-and-Arduino
An automated system for object detection and tracking using YOLOv9

link {https://www.youtube.com/watch?v=iEbMcPZ-qHQ}  ----> How it works this APP!!!
-- VERSIONS --
python 3.10.12
Ubuntu 22.04 Jetpack 6.0
CUDA 12.2.140
Release 5.15.136 - tegra
OpenCV 4.10.0-dev with CUDA 
torch 2.2.0a0+81ea7a4


THE MOST IMPORTANT RULE IS  that you need strong (very strong) GPU to run all the project..
I work with NVIDIA JETSON NANO ORIN and Arduino UNO 
Download the PanTiltTest.ino and adjust it to Arduino UNO
Download this documents: https://github.com/WongKinYiu/yolov9.git
Replace the detect.py from above repository with detect_pan.py from this repository.
Follow the instructions from above repository and Download these documents: 1. MyApp.desktop 2. myappteliko.py
Check the paths they're okay and first execute the MyApp.desktop
