# e0509_reach_pen_project_env_cfg.py

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from . import mdp as local_mdp  # 방금 수정한 rewards.py를 불러옵니다

@configclass
class ReachPenSceneCfg(InteractiveSceneCfg):
    """(공통) 무대 설정: 바닥, 빛, 펜"""
    
    # 1. 바닥
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # 2. 빛
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # 3. 펜 (목표물)
    pen = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pen",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/e0509_reach_pen_project/usd/pen.usd", 
            scale=(0.001, 0.001, 0.001),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.5), 
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    robot = None 

@configclass
class ActionsCfg:
    pass

# =========================================================
# ✅ [핵심] 관측(Observation) 설정
# =========================================================
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 1. 관절 각도
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # 2. 관절 속도
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # 3. 펜의 위치 (로봇 기준)
        object_position = ObsTerm(
            func=local_mdp.object_pos_rel, 
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("pen"),
            }
        )
        
        # 4. [NEW] 내 손끝(Sweet Spot)의 위치
        # 이걸 추가하면 AI가 자기 손 위치를 훨씬 빨리 깨닫습니다!
        ee_pos = ObsTerm(
            func=local_mdp.ee_position,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["rh_p12_rn_base"])
            }
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    reset_pen_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                # "x": (0.30, 0.45),   # 30~45cm 거리 (더 가깝게)
                # "y": (-0.10, 0.10),  # 좌우 범위를 좁혀서 정면 위주로
                # "z": (0.15, 0.35),   # 높이를 낮춤 (로봇이 뻗기 편한 높이)
                "x": (0.030, 0.045),   # 30~45cm 거리 (더 가깝게)
                "y": (-0.010, 0.010),  # 좌우 범위를 좁혀서 정면 위주로
                "z": (0.015, 0.035),   # 높이를 낮춤 (로봇이 뻗기 편한 높이)
                "roll": (-0.52, 0.52),  # 빨간축 기준 +-30도
                "pitch": (-0.52, 0.52), # 초록축 기준 +-30도
                "yaw": (-2.09, 2.09),   # 파란축 기준 +-120도
            },
            "velocity_range": {}, 
            "asset_cfg": SceneEntityCfg("pen"),
        },
    )

@configclass
class RewardsCfg:

    # 1. Action Rate Penalty: 이전 스텝과 현재 스텝의 액션 차이가 크면 벌점 (급격한 움직임 방지)
    action_rate = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.01  # 헤드뱅잉이 심하면 이 값을 -0.05 정도로 높이세요.
    )

    # 2. Joint Velocity Penalty: 관절 속도가 너무 빠르면 벌점
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-0.0001 # 관절이 너무 휘둘리는 것을 억제합니다.
    )   
    
    # 1. 거리 보상 (가까우면 점수)
    reaching_distance = RewTerm(
        func=local_mdp.object_ee_distance,
        weight=10.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["rh_p12_rn_base"]), 
            "object_cfg": SceneEntityCfg("pen")
        }
    )

    # 2. 방향 보상 (마주 보면 점수)
    reaching_orientation = RewTerm(
        func=local_mdp.pen_orientation_reward,
        weight=5.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["rh_p12_rn_base"]),
            "object_cfg": SceneEntityCfg("pen")
        }
    )

    
    # 4. 생존 보상
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class ReachPenEnvCfg(ManagerBasedRLEnvCfg):
    """(공통) 전체 환경 설정 틀"""
    scene: ReachPenSceneCfg = ReachPenSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (1.5, 1.5, 1.5)