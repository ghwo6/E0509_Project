import gymnasium as gym

# 1. 같은 폴더에 있는 환경 설정(레시피) 가져오기
from .env_cfg import E0509ReachPenEnvCfg, E0509ReachPenEnvCfg_v0, E0509ReachPenEnvCfg_v1

# 2. 에이전트 설정(학습 방법) 가져오기
from .agents.rsl_rl_ppo_cfg import PPORunnerCfg

# 3. 체육관(Gym) 메뉴판에 정식 등록!