import gymnasium as gym
import numpy as np
from features.transforms import Transforms

def get_env_features(env_name,max_episode_steps=2500):
	env = gym.make(env_name,render_mode='human',max_episode_steps=max_episode_steps)

	# Get number of actions from gym action space
	n_actions = env.action_space.n
	# Get the number of state observations
	state = env.reset()[0].shape
	state_space = (state[2], state[0], state[1])
	state_raw = np.zeros(state, dtype=np.uint8)
	# Preprocess state
	processed_state = Transforms.to_gray(state_raw)
	state_space = processed_state.shape

	return (env,state_space,n_actions)