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
#ifndef __DPRM_SERIAL_SERIAL_H__
#define __DPRM_SERIAL_SERIAL_H__

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "utils/timer.h"

namespace rm {

enum SerialType {
    SERIAL_TYPE_TTY_USB,
    SERIAL_TYPE_TTY_CH343USB,
    SERIAL_TYPE_TTY_THS,
    SERIAL_TYPE_TTY_ACM,
};

enum SerialStatus {
    SERIAL_STATUS_OK,
    SERIAL_STATUS_NOT_FOUND,
    SERIAL_STATUS_CHMOD_FAILED,
    SERIAL_STATUS_OPEN_FAILED,
    SERIAL_STATUS_SET_FILE_STATUS_FAILED,
    SERIAL_STATUS_GET_OLDTIO_FAILED,
    SERIAL_STATUS_SET_NEWTIO_FAILED,
    SERIAL_STATUS_NAME_ERROR,
    SERIAL_STATUS_CLOSE_FAILED,
    SERIAL_STATUS_READ_FAILED,
    SERIAL_STATUS_WRITE_FAILED,
    SERIAL_STATUS_INIT_HEAD_FAILED,
    SERIAL_STATUS_SOF_ERROR,
    SERIAL_STATUS_CRC_ERROR,
    SERIAL_STATUS_LENGTH_ERROR
};

SerialStatus
getSerialPortList(std::vector<std::string>& port_list, SerialType type = SERIAL_TYPE_TTY_USB);
SerialStatus closeSerialPort(int file_descriptor);
SerialStatus openSerialPort(
    int& file_descriptor,
    std::string name,
    int baudrate = 115200,
    char parity_bit = 'N',
    int data_bit = 8,
    int stop_bit = 1
);
SerialStatus restartSerialPort(
    int& file_descriptor,
    std::string name,
    int baudrate = 115200,
    char parity_bit = 'N',
    int data_bit = 8,
    int stop_bit = 1
);
SerialStatus readFromSerialPort(
    unsigned char* data,
    unsigned int length,
    int& file_descriptor,
    bool restart = false,
    std::string name = "/dev/ttyUSB0",
    int baudrate = 115200,
    char parity_bit = 'N',
    int data_bit = 8,
    int stop_bit = 1
);
SerialStatus writeToSerialPort(
    const unsigned char* data,
    unsigned int length,
    int& file_descriptor,
    bool restart = false,
    std::string name = "/dev/ttyUSB0",
    int baudrate = 115200,
    char parity_bit = 'N',
    int data_bit = 8,
    int stop_bit = 1
);

SerialStatus
initSerialHead(int& file_descriptor, size_t struct_size, const unsigned char sof = 0xA5);

} // namespace rm
#endif
