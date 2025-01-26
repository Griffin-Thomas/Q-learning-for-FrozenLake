import numpy as np
from shaped_frozen_lake import ShapedFrozenLake
from utils import parse_args


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
    
    # access flags from args
    is_slippery = args.is_slippery
    reward_shaping = args.reward_shaping
    
    # access hyperparameters from args
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon
    epsilon_decay = args.epsilon_decay
    min_epsilon = args.min_epsilon
    episodes = args.episodes
    max_steps = args.max_steps

    # initialize environment (FrozenLake with reward shaping custom class)
    env = ShapedFrozenLake(is_slippery=is_slippery, reward_shaping=reward_shaping)

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

            # update Q-table with Bellman equation
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