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
#ifndef __OPENRM_H__
#define __OPENRM_H__

#include <version.h>

#include <attack/attack.h>
#include <attack/deadlocker.h>
#include <attack/filtrate.h>
#include <attack/freshcenter.h>

#include <kalman/kalman.h>

#include <pointer/pointer.h>

#include <solver/solvepnp.h>
#include <solver/ternary.hpp>

#include <structure/cyclequeue.hpp>
#include <structure/slidestd.hpp>
#include <structure/speedqueue.hpp>
#include <structure/swapbuffer.hpp>

#include <structure/camera.hpp>
#include <structure/enums.hpp>
#include <structure/shm.hpp>
#include <structure/stamp.hpp>

#include <tensorrt/tensorrt.h>

#include <uniterm/uniterm.h>

#include <utils/delay.h>
#include <utils/print.h>
#include <utils/serial.h>
#include <utils/tf.h>
#include <utils/timer.h>

#include <video/video.h>

#endif
