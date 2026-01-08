# mdp/rewards.py

# ✅ 1. 이 줄이 없으면 Type Hint 에러가 납니다! (가장 중요)
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_rotate, quat_from_angle_axis

# 2. ManagerBasedRLEnv는 여기서만 불러오지만, 
# 맨 윗줄의 'annotations' 덕분에 아래 함수 정의에서 에러가 안 남
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_ee_distance(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    1. Sweet Spot(Z +0.105m)을 기준으로 거리를 잰다.
    2. Sweet Spot의 Z축과 펜의 Z축이 일치하는지 본다.
    3. (디버깅) 첫 실행 시 로봇의 관절 각도를 출력한다.
    """
    
    # [데이터 가져오기]
    robot: Articulation = env.scene[robot_cfg.name]
    pen: RigidObject = env.scene[object_cfg.name]
    
    # -------------------------------------------------------------
    # 🖨️ [디버깅] 초기 관절 각도 출력 (첫 번째 환경, 첫 스텝에서만)
    # -------------------------------------------------------------
    if env.common_step_counter == 0: 
        print("\n" + "="*50)
        print("🤖 [Robot Debug] Initial Joint Positions (Env 0)")
        # 첫 번째 로봇의 관절 각도 가져오기
        joint_pos_0 = robot.data.joint_pos[0].cpu().tolist()
        formatted_pos = [f"{x:.4f}" for x in joint_pos_0]
        print(f"   Joint Angles: {formatted_pos}")
        print("="*50 + "\n")

    # -------------------------------------------------------------
    # 🎯 Sweet Spot 위치 및 회전 계산
    # -------------------------------------------------------------
    # 1. 그리퍼 베이스(손목)의 위치와 회전
    ee_w = robot.data.body_state_w[:, robot_cfg.body_ids[0], :3]
    ee_quat_w = robot.data.body_state_w[:, robot_cfg.body_ids[0], 3:7]
    
    # 2. 오프셋 적용 (Base -> Sweet Spot : Z축으로 0.105m)
    offset_vec = torch.tensor([0.0, 0.0, 0.105], device=env.device).repeat(env.num_envs, 1)
    
    # 회전을 고려하여 오프셋을 더함
    sweet_spot_pos = ee_w + quat_rotate(ee_quat_w, offset_vec)

    # -------------------------------------------------------------
    # 📏 거리 보상 (Distance Reward)
    # -------------------------------------------------------------
    pen_pos = pen.data.root_pos_w
    distance = torch.norm(pen_pos - sweet_spot_pos, dim=-1)
    rew_dist = 1.0 / (1.0 + torch.square(distance)) # 가까울수록 점수 큼

    # -------------------------------------------------------------
    # 🧭 방향 보상 (Orientation Reward) - Z축 맞추기
    # -------------------------------------------------------------
    vec_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    sweet_spot_z_dir = quat_rotate(ee_quat_w, vec_z)

    pen_quat = pen.data.root_state_w[:, 3:7]
    pen_z_dir = quat_rotate(pen_quat, vec_z)

    dot_prod = torch.sum(sweet_spot_z_dir * pen_z_dir, dim=-1)
    rew_orient = torch.clamp((dot_prod + 1.0) / 2.0, min=0.0, max=1.0)

    # [최종 합산]
    total_reward = rew_dist + (rew_dist * rew_orient * 0.5)
    
    return total_reward

def pen_orientation_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    로봇 손(EE)과 펜(Object)의 회전 각도가 일치할수록 점수를 줍니다.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    pen: RigidObject = env.scene[object_cfg.name]
    
    ee_quat = robot.data.body_state_w[:, robot_cfg.body_ids[0], 3:7]
    pen_quat = pen.data.root_state_w[:, 3:7]
    
    quat_dot = torch.bmm(ee_quat.unsqueeze(1), pen_quat.unsqueeze(2)).squeeze()
    
    return torch.square(quat_dot).squeeze()