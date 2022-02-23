# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from .rank_filter import rank_filter


@rank_filter
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def collect_files(path, exts):
    file_paths = []
    for maindir, subdir, filename_list in os.walk(path):
        for filename in filename_list:
            file_path = os.path.join(maindir, filename)
            ext = os.path.splitext(file_path)[1]
            if ext in exts:
                file_paths.append(file_path)
    return file_paths

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]
def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names