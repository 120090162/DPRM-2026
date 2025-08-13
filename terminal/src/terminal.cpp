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
#include "terminal.h"
#include <fstream>
void readFile(const std::string& path, std::vector<double>& inputs) {
    std::ifstream fin;
    fin.open(path);
    if (!fin.is_open()) {
        std::cerr << "cannot open the file";
        return;
    }
    double tmp;
    while (fin >> tmp) {
        inputs.push_back(tmp);
    }
    if (inputs.size() % 3 != 0) {
        std::cerr << "num error";
    }
}
