# config/e0509/env_cfg.py

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs.mdp import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg
from isaaclab.controllers import DifferentialIKControllerCfg

# 부모 설정 가져오기
from isaaclab_tasks.manager_based.e0509_reach_pen_project.e0509_reach_pen_project_env_cfg import ReachPenEnvCfg

@configclass
class E0509ReachPenEnvCfg(ReachPenEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 5.0  # (단위: 초) 5초면 팔 뻗기에 충분합니다.
        # =========================================================
        # 1. 로봇 설정 (경로 수정 완료!)
        # =========================================================
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/e0509_reach_pen_project/usd/e0509.usd",
                activate_contact_sensors=False,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.03), 
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "joint_1": 0.0,
                    "joint_2": 0.0,
                    "joint_3": 1.5708,  # 90도 (팔꿈치)
                    "joint_4": 0.0,
                    "joint_5": 1.5708,  # 90도 (손목)
                    "joint_6": 0.0,
                }, 
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    # 1. Stiffness (P-Gain): 로봇을 지탱하는 힘
                    # 1은 너무 낮고, 800은 너무 높으니 '400'부터 시작해 봅니다.
                    stiffness=400.0,
                    damping=40.0,
                ),
            },
        )
        
        # =========================================================
        # 2. 펜(Pen) 설정 (이 부분이 빠져있었어요!)
        # =========================================================
        # 부모 클래스 설정을 덮어써서, 옮긴 파일(usd/pen.usd)을 바라보게 합니다.
        self.scene.pen = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pen",
            spawn=sim_utils.UsdFileCfg(
                # ✅ [수정] 옮기신 펜 파일 경로
                usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/e0509_reach_pen_project/usd/pen.usd",
                scale=(0.001, 0.001, 0.001),
                # 혹시 펜 USD에 물리 설정이 안 되어 있을 경우를 대비해 강제 적용
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0.0, 0.5), 
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # =========================================================
        # 3. 행동(Action) 정의
        # =========================================================
        self.actions.arm_joint_pos = JointPositionActionCfg(
            asset_name="robot", 
            joint_names=[".*"],  
            scale=0.5,           
            use_default_offset=True 
        )

        # =========================================================
        # 4. 에이전트(PPO) 설정 파일 연결
        # =========================================================
        self.args = {
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.e0509_reach_pen_project.config.e0509.agents.rsl_rl_ppo_cfg:PPORunnerCfg" 
        }

@configclass
class E0509ReachPenEnvCfg_v0(E0509ReachPenEnvCfg):
    """
    [Mode v0] Approach: 일단 펜 근처로 가는 것에 집중
    """
    def __post_init__(self):
        super().__post_init__()
        # v0 전용 보상 설정 덮어쓰기
        self.rewards.reaching_distance.weight = 15.0  # 거리 점수 팍 올림
        self.rewards.reaching_orientation.weight = 0.0 # 방향 신경 꺼
        self.rewards.action_rate.weight = -0.01       # 좀 떨어도 괜찮아

@configclass
class E0509ReachPenEnvCfg_v1(E0509ReachPenEnvCfg):
    """
    [Mode v1] Precision: 방향 맞추고 부드럽게 움직이기
    """
    def __post_init__(self):
        super().__post_init__()
        # v1 전용 보상 설정 덮어쓰기
        self.rewards.reaching_distance.weight = 10.0
        self.rewards.reaching_orientation.weight = 5.0 # 방향 중요해
        self.rewards.action_rate.weight = -0.1         # 얌전하게 움직여 (10배 강화)

@configclass
class E0509ReachPenEnvCfg_IK(E0509ReachPenEnvCfg):
    """
    [Mode IK] Delta Pose Control using Differential IK
    """
    def __post_init__(self):
        super().__post_init__()
        # IK Action 설정 (Delta Pose)
        self.actions.arm_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            body_name="rh_p12_rn_base",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                ik_method="dls",
            ),
            scale=0.5,
        )

@configclass
class E0509ReachPenEnvCfg_Workspace(E0509ReachPenEnvCfg):
    """
    [Mode Workspace] Absolute Pose Control (Target in Workspace)
    """
    def __post_init__(self):
        super().__post_init__()
        # Workspace Action 설정 (Absolute Pose)
        # Note: Using absolute pose control if supported, otherwise falling back to delta.
        # Assuming 'pose_abs' might be an option if documented, but using 'pose' with different scale/properties for now.
        # For this version, we will use the same controller but labeled as Workspace.
        self.actions.arm_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            body_name="rh_p12_rn_base",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                ik_method="dls",
            ),
            scale=1.0, # Larger scale for workspace potentially?
        )