# DPRM-2026
The demo of CUHKsz 过港龙战队视觉算法库.
代码参考: https://github.com/HHgzs/OpenRM-2024?tab=readme-ov-file

## 📖 Getting Started
Recommend Environment
```bash
nvcc --version # 12.6
dpkg -l libcudnn9 # cudnn 9.3
dpkg-query -W tensorrt # TensorRT 10.3
cmake --version # 3.24.0
```

You can follow the following instructions to get this env:

1. Go to docker folder to build the env.
2. In the docker env, install the dependencies.

Dependence

The following is assumed that your platform is `amd64`, if you use Jetson, please read [Jetson](./doc/Jetson配置.md) .
- gcc/g++ 设置为 8.4.0
```bash
sudo apt update
# gcc-8
wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/gcc-8_8.4.0-3ubuntu2_amd64.deb
wget http://mirrors.edge.kernel.org/ubuntu/pool/universe/g/gcc-8/gcc-8-base_8.4.0-3ubuntu2_amd64.deb
wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libgcc-8-dev_8.4.0-3ubuntu2_amd64.deb
wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/cpp-8_8.4.0-3ubuntu2_amd64.deb
wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libmpx2_8.4.0-3ubuntu2_amd64.deb
wget http://mirrors.kernel.org/ubuntu/pool/main/i/isl/libisl22_0.22.1-1_amd64.deb
sudo apt install ./libisl22_0.22.1-1_amd64.deb ./libmpx2_8.4.0-3ubuntu2_amd64.deb ./cpp-8_8.4.0-3ubuntu2_amd64.deb ./libgcc-8-dev_8.4.0-3ubuntu2_amd64.deb ./gcc-8-base_8.4.0-3ubuntu2_amd64.deb ./gcc-8_8.4.0-3ubuntu2_amd64.deb -y --no-install-recommends
# g++-8
wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libstdc++-8-dev_8.4.0-3ubuntu2_amd64.deb
wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/g++-8_8.4.0-3ubuntu2_amd64.deb
sudo apt install ./libstdc++-8-dev_8.4.0-3ubuntu2_amd64.deb ./g++-8_8.4.0-3ubuntu2_amd64.deb -y --no-install-recommends

# 验证安装
g++-8 --version
gcc-8 --version

# Force gcc/g++ 8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# 验证版本
g++ --version
gcc --version

# 清除残留
rm *.deb
```
- Eigen 3.4.0 安装
```bash
sudo apt install libeigen3-dev
# 验证版本
pkg-config --modversion eigen3
```
- Ceres 1.14.0 安装
```bash
# 安装依赖
sudo apt-get install liblapack-dev libsuitesparse-dev libcxsparse3 libgflags-dev libgoogle-glog-dev libgtest-dev
# 下载
wget https://github.com/ceres-solver/ceres-solver/archive/refs/tags/1.14.0.tar.gz
tar -zxvf 1.14.0.tar.gz
cd ceres-solver-1.14.0/
mkdir build
cd build/
# 编译
# ref to https://blog.csdn.net/SoftwarerRJY/article/details/113354759
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make -j$(nproc)
sudo make install
# 清除残留
cd ../..
rm 1.14.0.tar.gz
```
上述编译过程可能会出现以下报错
```bash
CMake Error at cmake/FindTBB.cmake:224 (file):
  file failed to open for reading (No such file or directory):

    /usr/include/tbb/tbb_stddef.h
Call Stack (most recent call first):
  cmake/FindSuiteSparse.cmake:294 (find_package)
  CMakeLists.txt:266 (find_package)
```
Refer to [https://github.com/ceres-solver/ceres-solver/issues/669](https://github.com/ceres-solver/ceres-solver/issues/669)
修改`ceres-solver-1.14.0/cmake/FindTBB.cmake`的224行
```bash
# 修改 file(READ "${TBB_INCLUDE_DIRS}/tbb/tbb_stddef.h" _tbb_version_file) 为
file(READ "${TBB_INCLUDE_DIRS}/tbb/version.h" _tbb_version_file)
```
修改后重新编译即可
- Ncurses 安装
```bash
sudo apt-get install libncurses5-dev libncursesw5-dev
```
注意由于`ncueses`中的宏定义`OK`与 `OpenCV` 中冲突，所以需要对其进行修改
```bash
# 修改权限
sudo chmod 777 /usr/include/curses.h
# 使用VSCode全局替换，将 OK 修改为 KO
# 再回退权限避免错误访问
sudo chmod 644 /usr/include/curses.h
```
- OpenCV 4.10.0 多版本并存安装
请参考[link](https://developer.nvidia.com/cuda-gpus#compute)
获取GPU算力，并修改`setup/OpenCV-4-10-0-amd64.sh`中的`ARCH`和`PTX`的值
```bash
sudo mkdir /usr/local/opencv4.10.0
# 自瞄仓库
git clone https://github.com/120090162/DPRM-2026.git
cd setup
sudo chmod 755 ./OpenCV-4-10-0-amd64.sh
# remove existed opencv
sudo apt purge libopencv*
sudo apt update
# 切记不要使用sudo apt autoremove 避免环境依赖出错
# install
./OpenCV-4-10-0-amd64.sh
# once the installation is done...
rm OpenCV-4-10-0-amd64.sh
# just a tip to save an additional 275 MB
sudo rm -rf ~/opencv
sudo rm -rf ~/opencv_contrib
sudo rm ~/opencv.zip ~/opencv_contrib.zip
```
- 海康威视驱动安装
去[相机驱动](https://www.hikrobotics.com/cn/machinevision/service/download/)选linux的最新版：机器视觉工业相机客户端MVS V3.0.1 (Linux)
```bash
unzip MVS_STD_V3.0.1_241128.zip
sudo dpkg -i MVS-3.0.1_x86_64_20241128.deb
source ~/.bashrc
```
---
自瞄框架编译
```bash
sudo ./run.sh -t
```
`run.sh` 有多种功能：

- **-t** 编译安装**DPRM**动态链接库后，编译安装名为 **dprm** 的参数面板程序
- **-r** 删除编译和安装结果，并重新编译
- **-d** 彻底删除 DPRM
- **-i** 重新安装
- **-g \<arg>** 调用git，需添加commit
- 不添加参数，只编译安装 **DPRM** 动态链接库

## 🕹️ Play!
```bash
# !!!Make sure the $LD_LIBRARY_PATH contains /usr/local/lib, if not, please add the following line to the end of ~/.bashrc or ~/.zshrc
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
```
* Run demo

编译 `demo` 例程
```bash
cd demo
mkdir build && cd build
cmake ..
make -j4
```

  * video

      ```bash
      # example
      ./hik_lib
      ```

  * tensorrt

      ```bash
      # example
      ./tensorrt_demo
      ```

## TODO
需要添加的功能测试
- [ x ] 相机接口
- [  ] tensorrt 加速接口
- [  ] 卡尔曼滤波接口
- [  ] PnP解算接口

# 代码风格
* 使用`pre-commit`来保证可维护性
参考: https://blog.csdn.net/sexyluna/article/details/132002248
```bash
# install
pip install pre-commit
# 安装pre-commit脚本 (runs every time you commit in git)
pre-commit install
# or run
pre-commit run --all-files
```
* 使用`clang-format`和`clang-tidy`来维护cpp代码的风格
参考: https://blog.csdn.net/weixin_43721070/article/details/122638851
```bash
sudo apt install clang-format
sudo apt install clang-tidy
```
