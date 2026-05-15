import gymnasium as gym
import multigrid.envs
from multigrid.envs.findgoal import FindGoalEnv
from multigrid.wrappers.external import TorchRLPettingZooWrapper
from torchrl.envs.libs import pettingzoo

# env = FindGoalEnv(room_size=7, num_rows=2, num_cols=3, num_agents= 2, render_mode="human")
# obs, info = env.reset()

env = FindGoalEnv(
    width=15,
    height=15,
    num_obstacles=6,
    agents=2,           # blue + red agent as in your image
    render_mode='human'
)
obs, info = env.reset(seed=0)

done = False
while not done:
    actions = {i: env.action_space[i].sample() for i in range(env.num_agents)}
    obs, rewards, terminations, truncations, info = env.step(actions)
    done = all(terminations.values()) or all(truncations.values())

env.close()