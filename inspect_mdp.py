import isaaclab.envs.mdp as mdp
import inspect

print("Attributes in isaaclab.envs.mdp:")
actions = [name for name in dir(mdp) if "Action" in name]
for a in actions:
    print(a)

print("\nAttributes in isaaclab.controllers (if importable):")
try:
    import isaaclab.controllers as controllers
    ctrls = [name for name in dir(controllers) if "Controller" in name]
    for c in ctrls:
        print(c)
except ImportError:
    print("Could not import isaaclab.controllers")
