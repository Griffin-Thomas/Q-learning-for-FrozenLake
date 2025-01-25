import argparse
import numpy as np
from shaped_frozen_lake import ShapedFrozenLake


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