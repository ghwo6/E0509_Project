# config/e0509/env_cfg.py

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs.mdp import JointPositionActionCfg

# 부모 설정 가져오기
from isaaclab_tasks.manager_based.e0509_reach_pen_project.e0509_reach_pen_project_env_cfg import ReachPenEnvCfg

@configclass
class E0509ReachPenEnvCfg(ReachPenEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # =========================================================
        # 1. 로봇 설정 (경로 수정 완료!)
        # =========================================================
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                # ✅ [수정] 옮기신 '평탄화(Flatten)'된 파일 경로
                usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/e0509_reach_pen_project/usd/e0509_flat.usd",
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
                    # ✅ [수정] 경고 로그 해결 (_sim 붙이기)
                    velocity_limit_sim=2.0, 
                    effort_limit_sim=87.0,
                    stiffness=800.0,
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
                scale=(1.0, 1.0, 1.0),
                # 혹시 펜 USD에 물리 설정이 안 되어 있을 경우를 대비해 강제 적용
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