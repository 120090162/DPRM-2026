#!/bin/bash
set -e

install_opencv () {
  NO_JOB=$(nproc)
  ARCH=8.9
  PTX="sm_89"
  
  echo "Installing OpenCV 4.10.0 using $NO_JOB threads for CUDA $ARCH with PTX $PTX on your system..."
  echo "It will take a very long time !"
  
  # refer to https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7
  # 依赖安装
  sudo apt update
  sudo apt install -y build-essential cmake unzip pkg-config git
  # Image I/O libs
  sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
  # Install basic codec libraries
  sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
  # Install GStreamer development libraries
  sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
  # Install additional codec and format libraries
  sudo apt install -y libxvidcore-dev libx264-dev libmp3lame-dev libopus-dev
  # Install additional audio codec libraries
  sudo apt install -y libmp3lame-dev libvorbis-dev
  # Install FFmpeg (which includes libavresample functionality)
  sudo apt install -y ffmpeg
  # Install video capture libraries and utilities
  sudo apt install -y libdc1394-dev libxine2-dev libv4l-dev v4l-utils
  # GTK lib for the graphical user functionalites coming from OpenCV highghui module
  sudo apt install -y libgtk-3-dev
  # Parallelism library C++ for CPU
  sudo apt install -y libtbb-dev
  # Optimization libraries for OpenCV
  sudo apt install -y libatlas-base-dev gfortran
  # Optional libraries:
  sudo apt install -y libhdf5-dev libprotobuf-dev protobuf-compiler
  sudo apt install -y libgoogle-glog-dev libgflags-dev

  # remove old versions or previous builds
  cd ~ 
  sudo rm -rf opencv*
  # download the latest version
  wget -O opencv.zip https://github.com/opencv/opencv/archive/4.10.0.zip 
  wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.10.0.zip 
  
  # unpack
  unzip opencv.zip 
  unzip opencv_contrib.zip 

  # Some administration to make life easier later on
  mv opencv-4.10.0 opencv
  mv opencv_contrib-4.10.0 opencv_contrib

  # set install dir
  cd ~/opencv
  mkdir build
  cd build
  
  # run cmake
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local/opencv4.10.0 \
  -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
  -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
  -D WITH_OPENGL=ON \
  -D CUDA_ARCH_BIN=${ARCH} \
  -D CUDA_ARCH_PTX=${PTX} \
  -D WITH_CUDA=ON \
  -D WITH_CUDNN=ON \
  -D WITH_CUBLAS=ON \
  -D ENABLE_FAST_MATH=ON \
  -D CUDA_FAST_MATH=ON \
  -D OPENCV_DNN_CUDA=ON \
  -D WITH_QT=OFF \
  -D WITH_OPENMP=ON \
  -D BUILD_TIFF=ON \
  -D WITH_FFMPEG=ON \
  -D WITH_GSTREAMER=ON \
  -D WITH_TBB=ON \
  -D BUILD_TBB=ON \
  -D BUILD_TESTS=OFF \
  -D WITH_EIGEN=ON \
  -D WITH_V4L=ON \
  -D WITH_LIBV4L=ON \
  -D WITH_PROTOBUF=ON \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D INSTALL_C_EXAMPLES=OFF \
  -D INSTALL_PYTHON_EXAMPLES=OFF \
  -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D BUILD_EXAMPLES=OFF \
  -D CMAKE_CXX_FLAGS="-march=native -mtune=native" \
  -D CMAKE_C_FLAGS="-march=native -mtune=native" ..
 
  make -j ${NO_JOB} 
  
  sudo make install
  sudo ldconfig
  
  # cleaning (frees 320 MB)
  make clean
  sudo apt-get update
  
  echo "Congratulations!"
  echo "You've successfully installed OpenCV 4.10.0 on your Nano"
}

cd ~

if [ -d ~/opencv/build ]; then
  echo " "
  echo "You have a directory ~/opencv/build on your disk."
  echo "Continuing the installation will replace this folder."
  echo " "
  
  printf "Do you wish to continue (Y/n)?"
  read answer

  if [ "$answer" != "${answer#[Nn]}" ] ;then 
      echo "Leaving without installing OpenCV"
  else
      install_opencv
  fi
else
    install_opencv
fi