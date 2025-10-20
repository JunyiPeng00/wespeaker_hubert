# Copyright (c) 2024 Hongji Wang (jijijiang77@gmail.com)
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


# from .s3prl import S3prlFrontend
try:
    from .whisper_encoder import whisper_encoder
    WHISPER_AVAILABLE = True
except ImportError:
    whisper_encoder = None
    WHISPER_AVAILABLE = False

from .get_hf_ssl_pruning import HuggingfaceFrontend

frontend_class_dict = {
    'fbank': None,
    # 's3prl': S3prlFrontend,
    'huggingface': HuggingfaceFrontend
}

if WHISPER_AVAILABLE:
    frontend_class_dict['whisper_encoder'] = whisper_encoder