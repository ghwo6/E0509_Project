# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

# 1. 보상 함수(mdp) 불러오기
from . import mdp

# 2. 설정(config) 폴더 불러오기 
# (이 줄이 실행되면서 config/e0509/__init__.py 안에 있는 gym.register가 자동으로 작동합니다!)
from .config import e0509