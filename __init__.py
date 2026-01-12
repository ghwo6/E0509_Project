# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

# 1. 보상 함수(mdp) 불러오기
import gymnasium as gym
from . import mdp
from .config.e0509 import agents
# 2. 설정(config) 폴더 불러오기 
# (이 줄이 실행되면서 config/e0509/__init__.py 안에 있는 gym.register가 자동으로 작동합니다!)
from .config import e0509

gym.register(
    id="Isaac-E0509-Reach-Pen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": e0509.E0509ReachPenEnvCfg_v0,
        "rsl_rl_cfg_entry_point": e0509.PPORunnerCfg,
    },
)

# ---------------------------------------------------------
#  [추가] v1 버전 등록 (Precision Mode)
# ---------------------------------------------------------
gym.register(
    id="Isaac-E0509-Reach-Pen-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": e0509.E0509ReachPenEnvCfg_v1,
        "rsl_rl_cfg_entry_point": e0509.PPORunnerCfg,
    },
)