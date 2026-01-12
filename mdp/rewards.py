# mdp/rewards.py

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_rotate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# =========================================================================
# 1. [관측용] 손끝(Sweet Spot)의 위치를 알려주는 함수
# =========================================================================
def ee_position(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    로봇 팔의 실제 끝부분(Sweet Spot)이 어디인지(x, y, z) 계산해서 알려줍니다.
    AI가 "내 손이 어디 있지?"라고 고민하지 않게 도와줍니다.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    
    # 1. 손목(Base)의 위치와 회전 가져오기
    # (config에서 지정한 'rh_p12_rn_base' 링크의 정보)
    ee_w = robot.data.body_state_w[:, robot_cfg.body_ids[0], :3]
    ee_quat_w = robot.data.body_state_w[:, robot_cfg.body_ids[0], 3:7]
    
    # 2. 오프셋 적용 (손목에서 Z축으로 +0.105m 떨어진 곳이 진짜 잡는 위치)
    offset_vec = torch.tensor([0.0, 0.0, 0.105], device=env.device).repeat(env.num_envs, 1)
    
    # 회전을 고려해서 오프셋 더하기
    sweet_spot_pos = ee_w + quat_rotate(ee_quat_w, offset_vec)
    
    # 3. (선택) 로봇 몸통(0,0,0)을 기준으로 한 상대 좌표로 변환
    # 절대 좌표보다 상대 좌표가 학습에 더 유리합니다.
    robot_base_pos = robot.data.root_pos_w
    return sweet_spot_pos - robot_base_pos


# =========================================================================
# 1-2. [관측용] 물체(펜)의 위치를 알려주는 함수 (NEW)
# =========================================================================
def object_pos_rel(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    로봇 베이스 기준, 목표물(펜)의 상대 위치를 계산합니다.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    pen: RigidObject = env.scene[object_cfg.name]

    # 로봇 베이스 위치 (보통 (0,0,0)이지만, 멀티 GPU 학습시 월드 좌표는 다를 수 있음)
    robot_base_pos = robot.data.root_pos_w
    
    # 펜의 위치
    pen_pos = pen.data.root_pos_w
    
    # 상대 위치 반환
    return pen_pos - robot_base_pos


# =========================================================================
# 2. [보상용] 거리 점수 (오프셋 적용됨)
# =========================================================================
def object_ee_distance(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    pen: RigidObject = env.scene[object_cfg.name]

    # 로봇 손 위치 계산 (위와 동일한 로직)
    ee_w = robot.data.body_state_w[:, robot_cfg.body_ids[0], :3]
    ee_quat_w = robot.data.body_state_w[:, robot_cfg.body_ids[0], 3:7]
    offset_vec = torch.tensor([0.0, 0.0, 0.105], device=env.device).repeat(env.num_envs, 1)
    sweet_spot_pos = ee_w + quat_rotate(ee_quat_w, offset_vec)

    # 펜 위치
    pen_pos = pen.data.root_pos_w
    
    # 거리 계산
    distance = torch.norm(pen_pos - sweet_spot_pos, dim=-1)
    
    # 점수 반환 (0 ~ 1)
    return 1.0 / (1.0 + torch.square(distance))


# =========================================================================
# 3. [보상용] 방향 점수 (마주 보면 만점)
# =========================================================================
def pen_orientation_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    pen: RigidObject = env.scene[object_cfg.name]

    # 회전값 가져오기
    ee_quat = robot.data.body_state_w[:, robot_cfg.body_ids[0], 3:7]
    pen_quat = pen.data.root_state_w[:, 3:7]

    # 파란색 선(Z축) 벡터
    vec_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)

    # 방향 계산
    ee_z_dir = quat_rotate(ee_quat, vec_z)
    pen_z_dir = quat_rotate(pen_quat, vec_z)

    # 내적 계산
    dot_prod = torch.sum(ee_z_dir * pen_z_dir, dim=1)

    # 반대 방향(-1)일 때 1점 (0.5 * (1 - (-1)) = 1.0)
    return 0.5 * (1.0 - dot_prod)