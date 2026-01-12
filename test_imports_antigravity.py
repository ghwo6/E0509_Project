try:
    from isaaclab.envs.mdp import DifferentialInverseKinematicsActionCfg
    print("Found DifferentialInverseKinematicsActionCfg in isaaclab.envs.mdp")
except ImportError:
    print("NOT Found DifferentialInverseKinematicsActionCfg in isaaclab.envs.mdp")

try:
    from isaaclab.controllers import DifferentialIKControllerCfg
    print("Found DifferentialIKControllerCfg in isaaclab.controllers")
except ImportError:
    print("NOT Found DifferentialIKControllerCfg in isaaclab.controllers")
    try:
        from isaaclab.control import DifferentialIKControllerCfg
        print("Found DifferentialIKControllerCfg in isaaclab.control")
    except ImportError:
        print("NOT Found DifferentialIKControllerCfg in isaaclab.control")

