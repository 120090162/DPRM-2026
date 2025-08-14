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
#include <unistd.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "terminal.h"

int main(int argc, char** argv) {
    int option;
    bool oscilloscope_flag = false;
    bool monitor_flag = false;
    std::vector<std::string> arg_strs;
    std::vector<std::string> key_name { "autoaim", "camsense", "radar" };

    while ((option = getopt(argc, argv, "dhimov")) != -1) {
        switch (option) {
            case 'd':
                rm::term_init();
                rm::dashboard(key_name);
                break;
            case 'h':
                std::cout << "Usage: " << argv[0] << " [-d] [-h] [-i] [-v] [-o <arg> <arg> ... ]"
                          << std::endl;
                break;
            case 'i':
                std::cout << "Hello, World!" << std::endl;
                break;
            case 'm':
                rm::term_init();
                rm::monitor(key_name);
                break;
            case 'v':
                std::cout << "DPRM version: " << DPRM_VERSION << std::endl;
                break;
            case 'o':
                oscilloscope_flag = true;
                break;
        }
    }

    if (optind < argc) {
        for (int i = optind; i < argc; i++) {
            arg_strs.push_back(argv[i]);
        }
    }

    if (oscilloscope_flag) {
        rm::term_init();
        rm::oscilloscope(key_name, arg_strs);
    }

    return 0;
}
