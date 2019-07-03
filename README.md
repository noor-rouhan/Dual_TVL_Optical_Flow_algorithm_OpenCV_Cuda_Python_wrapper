Python-Boost-OpenCV Converter and OpenCV CUDA C++ TVL-1 Opticalflow python wrapper.
==================

OpenCV has Dual TV1 Optical flow algorithm in C++ and Python but it doesn't have python wrapper for CUDA version/ GPU accelerated version for this algorithm. So this is just a modification from this code:
https://github.com/Algomorph/pyboostcvconverter

If you want to develop your custom OpenCV custom module follow the above link.

***IMPORTANT***
In order to compile the source properly I've made slight change in CMakeLists.txt file. In the find package section, replace the line find_package "(OpenCV COMPONENTS core REQUIRED)" like below: 

```python
#=============== Find Packages ====================================
## OpenCV
find_package(OpenCV REQUIRED)


```


Compatibility
-----------------
This code is compatible with OpenCV 2.X, 3.X, and 4.X.
This code supports Python 2.7 and Python 3.X. => You can pick one by passing `-DPYTHON_DESIRED_VERSION=3.X` or `=2.X` to cmake.

Compiling & Trying Out Sample Code
----------------------
1. Install CMake and/or CMake-gui (http://www.cmake.org/download/, ```sudo apt-get install cmake cmake-gui``` on Ubuntu/Debian)
2. Run CMake and/or CMake-gui with the git repository as the source and a build folder of your choice (in-source builds supported.) Choose desired generator, configure, and generate. Remember to set PYTHON_DESIRED_VERSION to 2.X for python 2 and 3.X for python 3.
3. Build (run the appropriate command ```make``` or ```ninja``` depending on your generator, or issue "Build All" on Windows+MSVC)
4. On *nix systems, ```make install``` run with root privileges will install the compiled library file. Alternatively, you can manually copy it to the pythonXX/dist-packages directory (replace XX with desired python version). On Windows+MSVC, build the INSTALL project.
5. Run python interpreter of your choice, issue the following commands:
```python
import numpy
import pbcvt as pb # custom module
import cv2 as cv 

image1 = cv.imread("test1.jpg",cv.IMREAD_GRAYSCALE)
image2 = cv.imread("test2.jpg",cv.IMREAD_GRAYSCALE)

optical_flow = pb.tvl1_optical_flow_cuda(image1,image2) #tvl1_optical_flow_cuda() python wrapper for OpenCV C++ Dual TVL1 optical flow algorithm with CUDA support. 

```
    
Credits
----------------
Original code: https://github.com/Algomorph/pyboostcvconverter

[Yati Sagade's example](https://github.com/yati-sagade/blog-content/blob/master/content/numpy-boost-python-opencv.rst).
