
from torchviz import make_dot
from dqn_agent import DQAgent
from features.get_env_features import get_env_features
from config.config import *
from features.transforms import Transforms
import torch



def main():
	env,state_space,n_actions = get_env_features(ENV_NAME)
	env.reset()
	obs = env.step(1)[0]
	state = Transforms.to_gray(obs)

	agent = DQAgent(replace_target_cnt=5000, env=env, state_space=state_space, 	action_space=n_actions, gamma=GAMMA,
					eps_strt=EPS_START, eps_end=EPS_END, eps_dec=EPS_DECAY, batch_size=BATCH_SIZE, lr=LR)

	X = torch.tensor(state).float().to('cpu')
	X = X.unsqueeze(0)
	y = agent.policy_net(X)


	make_dot(y.mean(), params=dict(list(agent.policy_net.named_parameters()))).render("rnn_torchviz", format="png")




if __name__=='__main__':
	main()