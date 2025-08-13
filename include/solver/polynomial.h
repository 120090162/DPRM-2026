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
#ifndef __OPENRM_SOLVER_POLYNOMIAL_H__
#define __OPENRM_SOLVER_POLYNOMIAL_H__

namespace rm {

double getPrediction(double x, double y);

void initFit(int x_max, int y_max);

void calculateFactors(std::vector<double>& inputs, bool isNeedR_2);

void getFitFactors();

double getR_2();

}; // namespace rm

#endif
