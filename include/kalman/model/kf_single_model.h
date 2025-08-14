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
#ifndef __DPRM_KALMAN_MODEL_KF_SINGLE_MODEL_H__
#define __DPRM_KALMAN_MODEL_KF_SINGLE_MODEL_H__

// [ x, y, z, theta, vx, vy]  [ x, y, z, theta]
// [ 0, 1, 2,   3,   4,  5 ]  [ 0, 1, 2,   3  ]

namespace KF_SM {

const int dimX = 6;
const int dimY = 4;

struct FuncA {
    double dt;
    template<class T>
    void operator()(T& A) {
        A = T::Identity();
        A(0, 4) = dt;
        A(1, 5) = dt;
    }
};

struct FuncH {
    template<class T>
    void operator()(T& H) {
        H = T::Zero();
        H(0, 0) = 1;
        H(1, 1) = 1;
        H(2, 2) = 1;
        H(3, 3) = 1;
    }
};

} // namespace KF_SM

#endif
