# DQN Reinforcement Learning in Atari Games



## Introduction
A DQN, or Deep Q-Network, approximates a state-value function in a Q-Learning framework with a neural network. In the Atari Games case, they take in several frames of the game as an input and output state values for each action as an output.

It is usually used in conjunction with Experience Replay, for storing the episode steps in memory for off-policy learning, where samples are drawn from the replay memory at random. Additionally, the Q-Network is usually optimized towards a frozen target network that is periodically updated with the latest weights every k steps (where k is a hyperparameter). The latter makes training more stable by preventing short-term oscillations from a moving target. The former tackles autocorrelation that would occur from on-line learning, and having a replay memory makes the problem more like a supervised learning problem.



## The Game
**Breakout** is a classic arcade video game that was first released by Atari in 1976. It is a simple yet challenging game that has become a popular choice for testing and developing reinforcement learning algorithms. The game's objective is to break a wall of bricks using a paddle and ball, which bounces off the walls and obstacles.

![The Game](reports/figures/game_frame.png "The Game")

### Objective
The primary objective of the Breakout game is to clear the screen of all the bricks. Players control a paddle located at the bottom of the screen, which can be moved left and right. A ball is launched from the paddle, and it bounces around the screen, colliding with bricks. The player must move the paddle to ensure that the ball doesn't fall off the screen. When the ball strikes a brick, the brick disappears, and the player scores points. The game continues until all the bricks are removed or the player loses all their lives.

### Complexity
While the concept of the Breakout game is straightforward, it presents several challenges for AI agents, particularly when using reinforcement learning. These challenges include:
 - **High-Dimensional State Space**: The game is typically played using pixel-based graphics, resulting in a high-dimensional state space that the AI agent must interpret. This makes learning an optimal policy more challenging.
 - **Sparse Rewards**: The rewards in Breakout are sparse, as they are primarily obtained by breaking bricks. It can take a significant number of actions to achieve a meaningful reward signal.
 - **Exploration vs. Exploitation**: Agents must strike a balance between exploring different actions to discover the best strategy and exploiting actions that have yielded positive results.
 - **Temporal Credit Assignment**: Determining which actions contributed to a successful or unsuccessful outcome is challenging due to the delayed nature of rewards in the game.



## DQN Algorithm
**Deep Q-Network (DQN)** is a deep reinforcement learning algorithm that combines neural networks with Q-learning to enable agents to make decisions in complex and high-dimensional environments. It was introduced by Volodymyr Mnih et al. in their 2015 paper "Human-level control through deep reinforcement learning."

### Components of the DQN Algorithm
- **Q-Network**: The core of the DQN is a neural network, often a deep convolutional neural network (CNN). This network is responsible for approximating the Q-function, denoted as Q(s, a), which estimates the expected cumulative reward of taking action 'a' in state 's'. The Q-network takes the current state as input and outputs Q-values for all possible actions.
- **Experience Replay**: DQN uses an experience replay buffer, which is a storage of past experiences (state, action, reward, next state). This buffer helps in breaking the temporal correlation of sequential experiences, making training more stable. During training, random batches of experiences are sampled from the replay buffer for updates.
- **Target Network**: To stabilize training, DQN employs two networks: the primary Q-network and a target network. The target network is a copy of the Q-network but with frozen parameters. The Q-network is periodically synchronized with the target network. This technique helps prevent the target Q-values from "moving" during training, improving stability.

### Q-Learning Update Equation
The core of the DQN algorithm is the Q-learning update equation, which is used to iteratively update the Q-values. It is based on the temporal difference (TD) error:

$$ Q(s, a) = Q(s, a) + \alpha * [r + \gamma * max(Q(s', a')) - Q(s, a)] $$

- $Q(s, a)$ is the Q-value for taking action $a$ in state $s$.
- $\alpha$ is the learning rate, determining the step size for the updates.
- $r$ is the immediate reward obtained after taking action $a$ in state $s$.
- $\gamma$ is the discount factor, which weighs future rewards.
- $max(Q(s', a'))$ represents the maximum Q-value for the next state $s$ considering all possible actions $a$.

### Training
 - **Initialization**: The Q-network and the target network are initialized with random weights. The experience replay buffer is also initialized.
 - **Exploration vs. Exploitation**: During training, the agent must balance exploration (trying new actions) and exploitation (choosing the best-known actions). This balance is often achieved using an ε-greedy policy, where the agent selects the best-known action with probability 1-ε and explores with probability ε.
 - **Experience Collection**: The agent interacts with the environment, collecting experiences in the form of (state, action, reward, next state).
 - **Experience Replay**: Periodically, a random batch of experiences is sampled from the replay buffer, and the Q-network is updated using the Q-learning update equation.
 - **Target Network Update**: The target network is updated to match the Q-network periodically to stabilize training.
 - **Convergence**: Training continues until the Q-values start converging towards the optimal values.



## Project Implementation

### Gym Implementation
Explain the steps taken to implement the DQN algorithm for the Breakout game in Gymnasium.

### Preprocessing
Discuss the preprocessing of game frames, including image resizing and color channel conversion.

![Preprocessing](reports/figures/preprocess.png "Preprocessing")

### The Network
Describe the neural network architecture used in the Q-network and the target network.

![The Network](reports/figures/rnn_torchviz.png "The Network")

Provide details on how experience replay is employed to stabilize training.


## Training and Results
Discuss the training process, including the choice of hyperparameters, training episodes, and exploration strategies.
Share the performance metrics used to evaluate the agent's progress during training.

![Results](reports/figures/scores_20230831080619.png "Results")

Present the results of the DQN project, including any improvements observed over time.

![Small Training](reports/gif/breakout_model_50_000.gif "Small Training")
![More Training](reports/gif/breakout_model_75_000.gif "More Training")
![Even More Training](reports/gif/breakout_model_100_000.gif "Even More Training")