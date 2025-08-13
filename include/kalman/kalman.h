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
#ifndef __OPENRM_KALMAN_H__
#define __OPENRM_KALMAN_H__

#include <kalman/filter/ekf.h>
#include <kalman/filter/kf.h>

#include <kalman/model/ekf_center_model.h>
#include <kalman/model/ekf_single_model.h>
#include <kalman/model/kf_single_model.h>

// #include <kalman/interface/antitopV1.h>
// #include <kalman/interface/antitopV2.h>
#include <kalman/interface/antitopV3.h>

// #include <kalman/interface/trackqueueV1.h>
// #include <kalman/interface/trackqueueV2.h>
#include <kalman/interface/trackqueueV3.h>
#include <kalman/interface/trackqueueV4.h>

// #include <kalman/interface/runeV1.h>
#include <kalman/interface/outpostV1.h>
#include <kalman/interface/outpostV2.h>
#include <kalman/interface/runeV2.h>
#include <kalman/interface/trajectoryV1.h>

#endif
