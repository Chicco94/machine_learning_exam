import numpy as np
from config.config import *
from dqn_agent import DQAgent
from features.get_env_features import get_env_features
from visualization.visualize import plot_durations


def test_model(start=0,end=100_000,count=100,render=True,num_eps=5):

	env,state_space,n_actions = get_env_features(ENV_NAME)

	test_results = []

	for i in range(start,end,int((end-start)/count)):
		model_name = f'breakout_model_{max(0,i-100)}'
		agent = DQAgent(replace_target_cnt=1, env=env, state_space=state_space, action_space=n_actions, model_name=model_name, gamma=GAMMA,
					eps_strt=EPS_START, eps_end=EPS_END, eps_dec=EPS_DECAY, batch_size=BATCH_SIZE, lr=LR)
		scores,steps = agent.play_games(num_eps=num_eps, render=render)
		test_results.append((np.mean(scores),np.mean(steps)))
		print(f'\n\tModel {model_name} avg. score: {np.mean(scores)} avg. steps: {np.mean(steps)}\n')
		with open('../reports/log.log','a') as data:
			data.write(f'{model_name};{np.mean(scores)};{np.mean(steps)}\n')

	plot_durations(test_results,show_result=True)


if __name__=='__main__':
	test_model()