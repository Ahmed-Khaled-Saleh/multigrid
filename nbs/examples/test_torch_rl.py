import gymnasium as gym
import multigrid.envs
from multigrid.wrappers.external import TorchRLPettingZooWrapper
from torchrl.envs.libs import pettingzoo

env = gym.make('MultiGrid-FindGoal-8x8-v0', agents=2, render_mode='human')
env = TorchRLPettingZooWrapper(env)
env = pettingzoo.PettingZooWrapper(
    env=env,
    return_state=False,
    group_map=None,
)

# TorchRL returns a TensorDict, not a tuple
tensordict = env.reset()
print(env)
# print(tensordict)

while not env.is_done():  # you may need to check TorchRL's done API
    tensordict = env.rand_action(tensordict)  # sample random actions
    tensordict = env.step(tensordict)         # returns a TensorDict
    print(tensordict)

env.close()


# tensordict = env.reset()

# while True:
#     # Sample actions via TorchRL's API
#     tensordict = env.rand_action(tensordict)
#     tensordict = env.step(tensordict)
    
#     # Check if done
#     if tensordict.get("done").any():
#         break

# env.close()