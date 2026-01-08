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
from . import mdp as local_mdp

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
            usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/e0509_reach_pen_project/pen.usd", 
            scale=(1.0, 1.0, 1.0),
            
            # ✅ [추가 1] 강제로 물리 속성(Rigid Body) 켜기
            # 이걸 넣으면 USD 파일에 설정이 없어도 강제로 적용됩니다.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, # "너는 이제부터 물체다!"
                disable_gravity=False,   # 중력 적용 (떨어지게 함)
            ),
            
            # ✅ [추가 2] 질량 부여 (펜이니까 가볍게 0.05kg)
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

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
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
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    
    reaching_distance = RewTerm(
        func=local_mdp.object_ee_distance,
        weight=10.0,
        params={
            # ✅ [수정] body_ids -> body_names 로 변경! (이게 정답입니다)
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper_rh_p12_rn_base"]), 
            "object_cfg": SceneEntityCfg("pen")
        }
    )

    reaching_orientation = RewTerm(
        func=local_mdp.pen_orientation_reward,
        weight=5.0,
        params={
            # ✅ [수정] 여기도 body_names로 변경!
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper_rh_p12_rn_base"]),
            "object_cfg": SceneEntityCfg("pen")
        }
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)

    
# ✅ 여기가 핵심입니다! (종료 조건 클래스)
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
    # ✅ 여기에 등록!
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (1.5, 1.5, 1.5)