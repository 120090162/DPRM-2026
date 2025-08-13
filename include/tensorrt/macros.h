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
#ifndef __MACROS_H
#define __MACROS_H

#include <NvInfer.h>

#ifdef API_EXPORTS
    #if defined(_MSC_VER)
        #define API __declspec(dllexport)
    #else
        #define API __attribute__((visibility("default")))
    #endif
#else

    #if defined(_MSC_VER)
        #define API __declspec(dllimport)
    #else
        #define API
    #endif
#endif // API_EXPORTS

#if NV_TENSORRT_MAJOR >= 8
    #define TRT_NOEXCEPT noexcept
    #define TRT_CONST_ENQUEUE const
#else
    #define TRT_NOEXCEPT
    #define TRT_CONST_ENQUEUE
#endif

#endif // __MACROS_H
