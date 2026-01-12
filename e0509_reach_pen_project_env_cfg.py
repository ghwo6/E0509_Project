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
            func=mdp.object_pos_rel,
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
                "x": (0.50, 0.70), "y": (-0.20, 0.20), "z": (0.33, 0.63),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {}, 
            "asset_cfg": SceneEntityCfg("pen"),
        },
    )

@configclass
class RewardsCfg:
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

    # 3. 동작 패널티 (너무 급하게 움직이면 감점)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    
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