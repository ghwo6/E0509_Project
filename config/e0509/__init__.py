import gymnasium as gym

# 1. 같은 폴더에 있는 환경 설정(레시피) 가져오기
from .env_cfg import E0509ReachPenEnvCfg

# 2. 에이전트 설정(학습 방법) 가져오기
from .agents.rsl_rl_ppo_cfg import PPORunnerCfg

# 3. 체육관(Gym) 메뉴판에 정식 등록!
gym.register(
    id="Isaac-E0509-Reach-Pen",                 # 우리가 부를 이름 (주문명)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # Isaac Lab의 기본 구동기
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": E0509ReachPenEnvCfg, # "이 메뉴는 이 레시피를 씁니다"
        "rsl_rl_cfg_entry_point": PPORunnerCfg,      # "학습은 이렇게 시킵니다"
    },
)