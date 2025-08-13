# Jetson环境配置
以下流程在Jetson Orin Nano 8gb 环境下测试，并且Jetpack的版本为6.2

- gcc/g++ 设置为 8.4.0
```bash
sudo apt update
wget http://ports.ubuntu.com/ubuntu-ports/pool/universe/g/gcc-8/gcc-8_8.4.0-3ubuntu2_arm64.deb
wget http://ports.ubuntu.com/ubuntu-ports/pool/universe/g/gcc-8/gcc-8-base_8.4.0-3ubuntu2_arm64.deb
wget http://ports.ubuntu.com/ubuntu-ports/pool/universe/g/gcc-8/libgcc-8-dev_8.4.0-3ubuntu2_arm64.deb
wget http://ports.ubuntu.com/ubuntu-ports/pool/universe/g/gcc-8/cpp-8_8.4.0-3ubuntu2_arm64.deb
wget http://ports.ubuntu.com/ubuntu-ports/pool/main/i/isl/libisl22_0.22.1-1_arm64.deb
wget http://ports.ubuntu.com/ubuntu-ports/pool/universe/g/gcc-8/libstdc++-8-dev_8.4.0-3ubuntu2_arm64.deb
wget http://ports.ubuntu.com/ubuntu-ports/pool/universe/g/gcc-8/g++-8_8.4.0-3ubuntu2_arm64.deb

sudo apt install ./gcc-8_8.4.0-3ubuntu2_arm64.deb ./gcc-8-base_8.4.0-3ubuntu2_arm64.deb ./libgcc-8-dev_8.4.0-3ubuntu2_arm64.deb ./cpp-8_8.4.0-3ubuntu2_arm64.deb ./libisl22_0.22.1-1_arm64.deb ./libstdc++-8-dev_8.4.0-3ubuntu2_arm64.deb ./g++-8_8.4.0-3ubuntu2_arm64.deb -y --no-install-recommends

# 验证安装
g++-8 --version
gcc-8 --version

# Force gcc 8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# 验证版本
g++ --version
gcc --version

# rm *.deb
rm *.deb
```
- Eigen 3.4.0 安装
```bash
sudo apt install libeigen3-dev
```
- Ceres 1.14.0 安装
```bash
sudo apt install liblapack-dev libsuitesparse-dev libcxsparse3 libgflags-dev libgoogle-glog-dev libgtest-dev

wget https://github.com/ceres-solver/ceres-solver/archive/refs/tags/1.14.0.tar.gz
tar -zxvf 1.14.0.tar.gz
cd ceres-solver-1.14.0/
mkdir build
cd build/

cmake ..
make -j$(nproc)
sudo make install

cd ../..
rm 1.14.0.tar.gz
```
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
参考：https://qengineering.eu/install-opencv-on-jetson-nano.html
参考只能使用opencv 4.10.0：https://forums.developer.nvidia.com/t/opencv-4-8-0-with-contrib-fails-on-jetson-orin-nx-jetpack-6-2/327595
请注意在安装opencv之前需要设置power mode为MAXN SUPER模式
```bash
sudo mkdir /usr/local/opencv4.10.0
# 自瞄仓库
git clone https://github.com/120090162/dp2026_auto_aim.git
cd setup
# check your memory first
free -m
# you need at least a total of 8.5 GB!
# if not, enlarge your swap space as explained in the guide
sudo chmod 755 ./OpenCV-4-10-0-arm64.sh
# remove existed opencv
sudo apt purge libopencv*
sudo apt update
# 切记不要使用sudo apt autoremove 避免环境依赖出错
# install
./OpenCV-4-10-0-arm64.sh
# once the installation is done...
rm OpenCV-4-10-0-arm64.sh
# just a tip to save an additional 275 MB
sudo rm -rf ~/opencv
sudo rm -rf ~/opencv_contrib
sudo rm ~/opencv.zip ~/opencv_contrib.zip
```
- 海康威视驱动安装

---
自瞄框架编译
```bash
sudo ./run.sh -t
```