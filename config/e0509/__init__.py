# config/e0509/__init__.py

import gymnasium as gym
from .env_cfg import E0509ReachPenEnvCfg

# ✅ PPO 에이전트 설정 파일 경로
agent_cfg_path = "isaaclab_tasks.manager_based.e0509_reach_pen_project.config.e0509.agents.rsl_rl_ppo_cfg:PPORunnerCfg"

gym.register(
    # ⚠️ 여기 이름이 명령어(--task)와 토씨 하나 틀리지 않고 똑같아야 합니다!
    id="Isaac-E0509-Reach-Pen-v0",  
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": E0509ReachPenEnvCfg,
        "rsl_rl_cfg_entry_point": agent_cfg_path, 
    },
)