#!/bin/bash

# Define the header text
header_text='/*
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
 */'

# Function to add header to files
add_header_to_files() {
    # 安全处理文件名中的空格和特殊字符
    while IFS= read -r -d $'\0' file; do
        # 备份原内容
        tmp_content=$(cat "$file")

        # 添加版权头
        echo "$header_text" > "$file"

        # 恢复内容（保留可能的 shebang 行）
        if [[ "$tmp_content" == \#!* ]]; then
            echo "$tmp_content" | head -n1 >> "$file"  # shebang
            echo "$tmp_content" | tail -n +2 >> "$file" # 其他内容
        else
            echo "$tmp_content" >> "$file"
        fi
    # 递归查找所有目标文件（包括子目录）
    done < <(find . -regex ".*/build/.*" -prune -o \( -name "*.h" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" \) -print0)
}

add_header_to_files
