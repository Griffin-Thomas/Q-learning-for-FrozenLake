import argparse
import gym
import numpy as np


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for Q-learning hyperparameters.

    This function uses argparse to parse hyperparameters such as learning rate, discount factor, 
    exploration rate, and others, allowing you to configure the Q-learning algorithm via command line.

    Parameters:
    - None

    Returns:
    - argparse.Namespace: Object containing parsed arguments
    """
    parser = argparse.ArgumentParser(description="Q-learning for FrozenLake")

    # hyperparameters
    parser.add_argument('--alpha', type=float, default=0.01, help='learning rate (alpha)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (gamma)')
    parser.add_argument('--epsilon', type=float, default=1.0, help='exploration rate (epsilon)')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='decay rate for epsilon')
    parser.add_argument('--min_epsilon', type=float, default=0.01, help='minimum value of epsilon')
    parser.add_argument('--episodes', type=int, default=10000, help='number of training episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='maximum steps per episode')

    return parser.parse_args()


def reward_shaping(state, reward, goal_position=(3, 3)):
    """
    Applies reward shaping to encourage behaviour toward the goal and penalize bad actions.
    
    Parameters:
    - state: The current state of the agent
    - reward: The original reward from the environment
        i.e. Reward schedule:
                Reach goal(G): +1
                Reach hole(H): 0
                Reach frozen(F): 0
    - goal_position: The position of the goal (default is (3, 3) for 4x4 FrozenLake)
        e.g. "4x4":[
                "SFFF",
                "FHFH",
                "FFFH",
                "HFFG"
                ] where S = start, F = frozen, H = hole, G = goal
    
    Returns:
    - Shaped reward
    """
    current_position = (state // 4, state % 4) 
    # Manhattan distance (sum of horizontal and vertical distance)
    distance_to_goal = abs(current_position[0] - goal_position[0]) + abs(current_position[1] - goal_position[1])
    
    # if the agent fell into a hole, penalize based on distance to goal
    if reward == 0:
        return -0.05 * distance_to_goal

    # if the agent reached the goal, no further shaping (reward is already 1)
    return reward


class ShapedFrozenLake(gym.Env):
    """
    A custom environment that modifies the reward structure of the FrozenLake environment in OpenAI Gym.
    This environment applies reward shaping to provide additional rewards when the agent approaches the goal.

    Parameters:
    - is_slippery (bool): If set to True, the environment will have slippery tiles, making movement less deterministic. If set to False, the environment is deterministic.
    - reward_shaping (bool): If set to True, additional rewards or penalties are provided to the agent based on proximity to the goal.

    Reward Structure:
    - Standard FrozenLake environment provides a reward of 1 for reaching the goal and 0 otherwise.
    - ShapedFrozenLake gives additional intermediate rewards or penalties for steps that bring the agent closer to the goal.

    Example Usage:
    env = ShapedFrozenLake(is_slippery=True, reward_shaping=True)
    state = env.reset()
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    """
    def __init__(self, is_slippery=True, reward_shaping=True):
        super(ShapedFrozenLake, self).__init__()
        self.env = gym.make('FrozenLake-v1', is_slippery=is_slippery)
        self.reward_shaping = reward_shaping
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        state, reward, done, _, info = self.env.step(action)

        # apply reward shaping
        if self.reward_shaping:
            # apply additional reward or penalty for proximity to the goal
            reward = reward_shaping(state, reward)    

        return state, reward, done, info


def train_agent() -> None:
    """
    Trains the Q-learning agent on the FrozenLake environment.

    This function initializes the environment, sets up the Q-table, and trains the agent over 
    multiple episodes using the Q-learning algorithm. It updates the Q-table with each step and 
    decays the epsilon value to balance exploration and exploitation.

    Parameters:
    - None

    Returns:
    - None
    """
    args = parse_args()
    
    # access hyperparameters from args
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon
    epsilon_decay = args.epsilon_decay
    min_epsilon = args.min_epsilon
    episodes = args.episodes
    max_steps = args.max_steps

    # initialize environment (FrozenLake with reward shaping custom class)
    env = ShapedFrozenLake(is_slippery=False, reward_shaping=False)

    # initialize Q-table
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Q-learning algorithm
    for episode in range(episodes):
        state, _ = env.reset() # reset environment at the start of each episode
        done = False
        total_reward = 0
        
        for _ in range(max_steps):
            # exploration vs exploitation
            if np.random.uniform(0, 1) < epsilon:
                # explore: choose a random action
                action = env.action_space.sample()
            else:
                # exploit: choose the best action based on Q-table
                action = np.argmax(q_table[state])

            # take the action and observe the next state and reward
            next_state, reward, done, _ = env.step(action)

            # update Q-table
            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

            state = next_state
            total_reward += reward
            
            if done:
                break

        # decay epsilon for exploration vs exploitation balance, but stop at min_epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # the total reward per 100 episodes
        if episode % 100 == 0:
            print(f"episode {episode}: total reward = {total_reward}, epsilon = {epsilon}")

    print("Training completed!")

# let's run it
train_agent()