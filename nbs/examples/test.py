import gymnasium as gym
import multigrid.envs

env = gym.make('MultiGrid-RedBlueDoors-8x8-v0', agents=2, render_mode='human')

observations, infos = env.reset()
while not env.unwrapped.is_done():
   actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
   print(actions)
   observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()