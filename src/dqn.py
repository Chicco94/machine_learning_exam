import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):

	def __init__(self, input_dim, output_dim, model_name='model', env_name='BreakoutDeterministic'):
		super(DQN, self).__init__()
		self.input_dim = input_dim
		channels, _, _ = input_dim

		# 3 conv layers, all with relu activations, first one with maxpool
		self.l1 = nn.Sequential(
			nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=2),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
		)

		# Calculate output dimensions for linear layer
		conv_output_size = self.conv_output_dim()
		lin1_output_size = 512

		# Two fully connected layers with one relu activation
		self.l2 = nn.Sequential(
			nn.Linear(conv_output_size, lin1_output_size),
			nn.ReLU(),
			nn.Linear(lin1_output_size, output_dim)
		)

		# Save filename for saving model
		self.model_name = model_name
		self.env_name = env_name

	# Calulates output dimension of conv layers
	def conv_output_dim(self):
		x = torch.zeros(1, *self.input_dim)
		x = self.l1(x)
		return int(np.prod(x.shape))

	# Performs forward pass through the network, returns action values
	def forward(self, x):
		x = self.l1(x)
		x = x.view(x.shape[0], -1)
		actions = self.l2(x)

		return actions

	def save(self,additional_info=''):
		torch.save(self.state_dict(), f'models/{self.model_name}{additional_info}.pth')
		t = open('models/tap.tap','w')
		t.close()

	def load(self):
		self.load_state_dict(torch.load(f'../models/BreakoutDeterministic/{self.model_name}.pth'))
		self.eval()