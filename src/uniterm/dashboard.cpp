/*
 * Copyright (c) 2026, Cuhksz DragonPass. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <curses.h>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include "structure/shm.hpp"
#include "uniterm/uniterm.h"

static std::map<int, rm::MsgNum> MsgMap;

static void get_printxy(int& x, int& y, int rows, int cols, int index) {
    x = index / cols;
    y = index % cols;
}

int rm::term_hash(const char* __cstr) {
    unsigned long __hash = 5381;
    int __c;
    while ((__c = *__cstr++))
        __hash = ((__hash << 5) + __hash) + __c;
    return __hash;
}

void rm::term_init() {
    initscr(); // 初始化并创建标准屏幕窗口
    cbreak(); // 设置行缓冲模式，无需回车键
    noecho(); // 关闭输入字符的回显功能
    nodelay(stdscr, TRUE); // 设置getchar()非阻塞模式
    curs_set(0); // 隐藏光标
    start_color(); // 使用颜色的前置函数

    // 初始化颜色对，即前景色与背景色
    init_pair(1, COLOR_WHITE, COLOR_RED);
    init_pair(2, COLOR_WHITE, COLOR_GREEN);
    init_pair(3, COLOR_WHITE, COLOR_BLUE);
    init_pair(4, COLOR_WHITE, COLOR_YELLOW);

    init_pair(5, COLOR_GREEN, COLOR_BLACK);
    init_pair(6, COLOR_YELLOW, COLOR_BLACK);
    init_pair(7, COLOR_RED, COLOR_BLACK);
    init_pair(8, COLOR_BLUE, COLOR_BLACK);
}

void rm::dashboard(std::vector<std::string>& key_name) {
    std::vector<MsgNum*> msg_num(key_name.size(), nullptr);
    std::vector<MsgStr*> msg_str(key_name.size(), nullptr);

    for (size_t i = 0; i < key_name.size(); i++) {
        std::string num_shm_name = NumMsgShmKey + key_name[i];
        msg_num[i] = SharedMemory<MsgNum>(num_shm_name, NumShmLen);

        std::string str_shm_name = StrMsgShmKey + key_name[i];
        msg_str[i] = SharedMemory<MsgStr>(str_shm_name, StrShmLen);
    }

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        for (size_t i = 0; i < key_name.size(); i++) {
            for (size_t j = 0; j < rm::NumShmLen; j++) {
                if (msg_num[i][j].name[0] == '\0')
                    break;
                int hash = term_hash(msg_num[i][j].name);
                MsgMap[hash] = msg_num[i][j];
            }
        }

        int index = 0, rows, cols, x, y;

        erase();
        getmaxyx(stdscr, rows, cols);

        for (auto it = MsgMap.begin(); it != MsgMap.end(); it++) {
            x = (index / rows) * 25;
            y = index % rows;
            index++;

            std::string numstr;
            switch (it->second.type) {
                case 'i':
                    numstr = std::to_string(it->second.num.i);
                    break;
                case 'f':
                    numstr = std::to_string(it->second.num.f);
                    break;
                case 'd':
                    numstr = std::to_string(it->second.num.d);
                    break;
                case 'c':
                    numstr = it->second.num.c;
                    break;
            }
            if (numstr.size() > 8)
                numstr = numstr.substr(0, 8);

            mvprintw(y, x, it->second.name);
            attron(COLOR_PAIR(8));
            mvprintw(y, x + 16, numstr.c_str());
            attroff(COLOR_PAIR(8));
        }

        for (size_t i = 0; i < key_name.size(); i++) {
            for (size_t j = 0; j < rm::NumShmLen; j++) {
                if (msg_str[i][j].str[0] == '\0')
                    break;
                std::string str = msg_str[i][j].str;
                x = cols - str.size() - 1;
                y = j;

                rm::MSG type = static_cast<rm::MSG>(msg_str[i][j].type);
                switch (type) {
                    case rm::MSG_NOTE:
                        attron(COLOR_PAIR(8));
                        mvprintw(y, x, str.c_str());
                        attroff(COLOR_PAIR(8));
                        break;
                    case rm::MSG_OK:
                        attron(COLOR_PAIR(2));
                        mvprintw(y, x, str.c_str());
                        attroff(COLOR_PAIR(2));
                        break;
                    case rm::MSG_WARNING:
                        attron(COLOR_PAIR(4));
                        mvprintw(y, x, str.c_str());
                        attroff(COLOR_PAIR(4));
                        break;
                    case rm::MSG_ERROR:
                        attron(COLOR_PAIR(1));
                        mvprintw(y, x, str.c_str());
                        attroff(COLOR_PAIR(1));
                        break;
                }
            }
        }

        refresh();
    }
}
