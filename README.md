# E0509 로봇 팔 펜 잡기 프로젝트

<!-- 파일경로는 다음과 같습니다. 어디든 IsaacLab경로기준 (Isaaclab_Loc)이라 적겠습니다. -->

    (Isaaclab_Loc)/source/isaaclab_tasks/isaaclab_tasks/manager_based/e0509_reach_pen_project

<!-- 도커안에서 실행 -->

    <!-- ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-E0509-Reach-Pen --num_envs=1 -->
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task="Isaac-E0509-Reach-Pen-v0" --num_envs=1 --headless

<!-- RL -->

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-E0509-Reach-Pen --max_iterations=10000 --headless

<!-- 학습 이어서

     ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-E0509-Reach-Pen --max_iterations=5000 --headless --resume --num_envs=2048 -->

<!-- 학습 이어서 -->

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task="Isaac-E0509-Reach-Pen-v0" \
    --num_envs=2048 \
    --max_iterations=5000 \
    --headless
