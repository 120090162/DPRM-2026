#!/bin/bash
# 定义颜色变量
blue="\033[1;34m"
yellow="\033[1;33m"
red="\033[1;31m"
reset="\033[0m"
# 检查总编译行数
include_count=$(find include -type f \( -name "*.h"  -o -name "*.cuh" \) -exec cat {} \; | wc -l)
src_count=$(find src -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.txt" \) -exec cat {} \; | wc -l)
cuda_count=$(find cuda/src cuda/include -type f \( -name "*.cuh" -o -name "*.cu" -o -name "*.h" -o -name "*.txt" \) -exec cat {} \; | wc -l)
terminal_count=$(find terminal/src terminal/include -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.txt" \) -exec cat {} \; | wc -l)
daheng_count=$(find include/video/daheng -type f \( -name "*.h" \) -exec cat {} \; | wc -l)

total=$((include_count + src_count + cuda_count + terminal_count - daheng_count))
# 获取最大线程数
max_threads=$(cat /proc/cpuinfo | grep "processor" | wc -l)

# 检查是否存在 build 目录，如果不存在则创建
if [ ! -d "build" ]; then
    mkdir build
fi
# 命令行参数处理
while getopts ":ritdg:" opt; do
    case $opt in
        # 重新构建
        r)
            echo -e "${yellow}<<<--- rebuild --->>>\n${reset}"
            cd build
            sudo make uninstall
            cd ..
            sudo rm -rf build
            mkdir build
            shift
            ;;
        # 重新安装
        i)
            echo -e "${yellow}<<<--- reinstall --->>>\n${reset}"
            cd build
            sudo make uninstall
            cd ..
            shift
            ;;
        # 构建终端应用
        t)
            echo -e "${yellow}<<<--- terminal --->>>\n${reset}"
            cd build
            cmake ..
            sudo make install
            sudo rm /usr/lib/libdprm_*
            sudo ln -s /usr/local/lib/libdprm_* /usr/lib
            cd ../terminal
            if [ ! -d "build" ]; then
                mkdir build
            fi
            cd build
            cmake ..
            make -j "$max_threads"
            make -j "$max_threads"
            sudo make install
            cd ../..
            exit 0
            shift
            ;;
        # 删除项目
        d)
            echo -e "${yellow}\nThe following files and directories will be deleted:${reset}"
            sudo find "$(pwd)" -maxdepth 1 -name "build"
            sudo find /usr/local/ -name "*dprm*"

            echo -e "${red}Are you sure you want to delete dprm? (y/n): ${reset}"
            read answer
            if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
                echo -e "${yellow}\n<<<--- delete --->>>\n${reset}"
                sudo find "$(pwd)" -maxdepth 1 -name "build" -exec rm -rf {} +
                sudo find /usr/local/ -name "*dprm*" -exec rm -rf {} +
            else
                echo -e "${yellow}Deletion operation canceled${reset}"
            fi
            exit 0
            ;;
        # git操作
        g)
            git_message=$OPTARG
            echo -e "${yellow}\n<--- Git $git_message --->${reset}"
            git pull
            git add -A
            git commit -m "$git_message"
            git push
            exit 0
            shift
            ;;
        # 未知参数
        \?)
            echo -e "${red}\n--- Unavailable param: -$OPTARG ---\n${reset}"
            ;;
        # 缺少参数值
        :)
            echo -e "${red}\n--- param -$OPTARG need a value ---\n${reset}"
            ;;
    esac
done



echo -e "${yellow}\n<<<--- start cmake --->>>\n${reset}"
cd build
cmake ..

echo -e "${yellow}\n<<<--- start make --->>>\n${reset}"
make -j "$max_threads"

echo -e "${yellow}\n<<<--- start install --->>>\n${reset}"
sudo make install
sudo rm /usr/lib/libdprm_*
sudo ln -s /usr/local/lib/libdprm_* /usr/lib

echo -e "${yellow}\n<<<--- Total Lines --->>>${reset}"
echo -e "${blue}           $total${reset}"

echo -e "${yellow}\n<<<--- Welcome DPRM --->>>\n${reset}"
