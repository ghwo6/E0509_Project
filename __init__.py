# source/isaaclab_tasks/isaaclab_tasks/manager_based/e0509_reach_pen_project/__init__.py

import gymnasium as gym

# ✅ 수정된 부분: config 폴더를 통해 정확한 위치를 지정합니다.
from .config import e0509

# v0 등록 (기초 훈련용)
if "Isaac-E0509-Reach-Pen-v0" not in gym.envs.registry:
    gym.register(
        id="Isaac-E0509-Reach-Pen-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": e0509.E0509ReachPenEnvCfg_v0,
            "rsl_rl_cfg_entry_point": e0509.PPORunnerCfg,
        },
    )

# v1 등록 (Precision Mode)
if "Isaac-E0509-Reach-Pen-v1" not in gym.envs.registry:
    gym.register(
        id="Isaac-E0509-Reach-Pen-v1",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": e0509.E0509ReachPenEnvCfg_v1,
            "rsl_rl_cfg_entry_point": e0509.PPORunnerCfg,
        },
    )

# IK Mode 등록 (Delta Controller)
if "Isaac-E0509-Reach-Pen-IK-v0" not in gym.envs.registry:
    gym.register(
        id="Isaac-E0509-Reach-Pen-IK-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": e0509.E0509ReachPenEnvCfg_IK,
            "rsl_rl_cfg_entry_point": e0509.PPORunnerCfg,
        },
    )

# Workspace Mode 등록 (Absolute/Workspace Controller)
if "Isaac-E0509-Reach-Pen-Workspace-v0" not in gym.envs.registry:
    gym.register(
        id="Isaac-E0509-Reach-Pen-Workspace-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": e0509.E0509ReachPenEnvCfg_Workspace,
            "rsl_rl_cfg_entry_point": e0509.PPORunnerCfg,
        },
    )