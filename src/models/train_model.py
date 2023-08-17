from config.config import *
from dqn_agent import DQAgent
from features.get_env_features import get_env_features
from visualization.visualize import plot_durations

def train_model(num_eps,model_name='breakout_model'):

	env,state_space,n_actions = get_env_features(ENV_NAME)

	agent = DQAgent(replace_target_cnt=5000, env=env, state_space=state_space, action_space=n_actions, model_name=model_name, gamma=GAMMA,
					eps_strt=EPS_START, eps_end=EPS_END, eps_dec=EPS_DECAY, batch_size=BATCH_SIZE, lr=LR)

	episode_durations = agent.train(num_eps=num_eps)

	print('Training complete')
	plot_durations(episode_durations,show_result=True)


if __name__=='__main__':
	train_model(100_000)