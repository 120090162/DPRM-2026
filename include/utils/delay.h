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
#ifndef __OPENRM_UTILS_DELAY_H__
#define __OPENRM_UTILS_DELAY_H__

namespace rm {

double getFlyDelay(
    double& yaw,
    double& pitch,
    const double speed,
    const double target_x,
    const double target_y,
    const double target_z,
    const double head_dx,
    const double head_dy,
    const double head_dz,
    const double barrel_dx,
    const double barrel_dy,
    const double barrel_dz,
    const double barrel_yaw,
    const double barrel_pitch
);

double getFlyDelay(
    double& yaw,
    double& pitch,
    const double speed,
    const double target_x,
    const double target_y,
    const double target_z
);

double getRotateDelay(const double current_yaw, const double target_yaw);

} // namespace rm

#endif
