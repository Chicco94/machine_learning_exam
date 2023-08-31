import matplotlib.pyplot as plt
import torch
from datetime import datetime
import numpy as np

def plot_durations(episode_durations,show_result=False):
	plt.figure(1)
	durations_t = torch.tensor(episode_durations, dtype=torch.float)
	if show_result:
		plt.title('Result')
	else:
		plt.clf()
		plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Duration')
	plt.plot(durations_t.numpy())
	# Take 100 episode averages and plot them too
	if len(durations_t) >= 100:
		means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(means.numpy())
	plt.savefig('../reports/figures/results_{datetime.now().strftime("%Y%m%d%H%M%S")}.png')
	plt.pause(0.001)  # pause a bit so that plots are updated
	plt.show()


def plot_scores(episode_scores,epochs=None):
	plt.figure(1)
	scores_t = torch.tensor(episode_scores, dtype=torch.float)
	plt.title('Scores')
	plt.xlabel('Epoch')
	plt.ylabel('Score')
	if epochs is not None:plt.plot(epochs,scores_t.numpy())
	else: plt.plot(scores_t.numpy())
	# Take 10 episode averages and plot them too
	if len(scores_t) >= 20:
		means = scores_t.unfold(0, 20, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(19), means))
		if epochs is not None: plt.plot(epochs,means.numpy())
		else: plt.plot(means.numpy())
	plt.savefig(f'../reports/figures/scores_{datetime.now().strftime("%Y%m%d%H%M%S")}.png')
	plt.show()