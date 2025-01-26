# Q-learning for FrozenLake

This repository contains an implementation of the **Q-learning** reinforcement learning algorithm applied to the **FrozenLake** environment from OpenAI Gym. The agent learns to navigate through a grid-based world to reach the goal while avoiding holes, using **Q-learning** to optimize its strategy.

## Game Visualization
Here is a gif of what the **FrozenLake** game looks like in action:
<p align="center">
  <img src="./assets/frozen_lake.gif" alt="Game gif" />
</p>

## Project Overview

**FrozenLake** is a simple grid-based environment where an agent (a "robot") starts at a designated starting position and aims to reach the goal while avoiding obstacles (holes in the ice). The environment has discrete states and actions, making it a suitable candidate for **Q-learning** to be applied.

This implementation demonstrates how an agent learns optimal actions over episodes (a complete run through an environment, from the start to the goal or until the agent reaches a terminal state i.e. falls through a hole in the ice) using **Q-learning**. The **Q-table** is updated (an iteration) after each action based on rewards received.

## Key Concepts

- **Q-learning**: A reinforcement learning algorithm that helps an agent learn the value of actions in different states, ultimately learning a policy that maximizes its total reward. More detail on [Wikipedia](https://en.wikipedia.org/wiki/Q-learning).
- **Q-table**: A table (or matrix) that stores the Q-values of state-action pairs. Each row corresponds to a specific state in the environment, and each column corresponds to a specific action. Hence, each cell represents the **Q-value** of the agent taking a specific action in a specific state.
- **State Space**: The set of all possible states in the environment. In FrozenLake, the agent moves on a grid, and the state is the position in the grid.
- **Action Space**: The possible actions the agent can take, i.e., moving up, down, left, or right.
- **Exploration vs. Exploitation**: The agent must balance exploring new actions (randomly) and exploiting known good actions (according to its Q-table).
- **Q-value**: An estimate of the total (or future) reward the agent can expect to receive from that state-action pair, assuming it follows the optimal policy from there onward. Over time, the agent updates these values to improve its policy (the strategy for choosing actions). It is updated using the **Bellman equation**.

--- 

The **Bellman equation** used to update the **Q-value**:

![Bellman equation](./assets/bellman.svg)

<p align="center"><a href="https://en.wikipedia.org/wiki/Q-learning#Algorithm" target="_blank">Sourced from Wikipedia - Q-learning Algorithm</a></p>

## Requirements

Ensure you have the following Python libraries installed:

- `gym==0.26.2`
- `numpy==1.26.4`

To install these dependencies, run:

```bash
pip install -r requirements.txt
```

### Running the Q-learning Algorithm

From the root directory, this will run the training with the default hyperparameter values:
```bash
> python -m q_learning.main
```

Note that you can always run the following to see which arguments are supported:
```bash
> python -m q_learning.main -h

usage: main.py [-h] [--is_slippery IS_SLIPPERY] [--reward_shaping REWARD_SHAPING] [--alpha ALPHA] [--gamma GAMMA]
               [--epsilon EPSILON] [--epsilon_decay EPSILON_DECAY] [--min_epsilon MIN_EPSILON] [--episodes EPISODES]
               [--max_steps MAX_STEPS]

Q-learning for FrozenLake

options:
  -h, --help            show this help message and exit
  --is_slippery IS_SLIPPERY
                        If True, environment is slippery.
  --reward_shaping REWARD_SHAPING
                        If True, reward shaping is applied.
  --alpha ALPHA         learning rate (alpha)
  --gamma GAMMA         discount factor (gamma)
  --epsilon EPSILON     exploration rate (epsilon)
  --epsilon_decay EPSILON_DECAY
                        decay rate for epsilon
  --min_epsilon MIN_EPSILON
                        minimum value of epsilon
  --episodes EPISODES   number of training episodes
  --max_steps MAX_STEPS
                        maximum steps per episode
```

Here is an example:
```bash
> python -m q_learning.main --is_slippery n --reward_shaping n --alpha 0.01 --gamma 0.99 --epsilon 1.0 --epsilon_decay 0.995 --min_epsilon 0.01 --episodes 10000 --max_steps 1000
```

## Hyperparameters

The following hyperparameters are used in the Q-learning algorithm to control the learning process:

- **Alpha (learning rate)** - `alpha`: 
  - Controls how much the Q-values are updated after each action. A higher value means the agent will update its Q-values more aggressively.
  - Typical range: [0.0, 1.0] - note that this is a closed interval.

- **Gamma (discount factor)** - `gamma`: 
  - Determines the importance of future rewards. A value of 0 means the agent only cares about immediate rewards, while a value close to 1 means it values future rewards highly.
  - Typical range: [0.0, 1.0]

- **Epsilon (exploration rate)** - `epsilon`:
  - Controls the exploration vs exploitation trade-off. A higher value encourages the agent to explore more by choosing random actions, while a lower value encourages exploitation of known actions.
  - Typical range: [0.0, 1.0]
  - Can decay over time as the agent becomes more confident in its policy.

- **Epsilon Decay (optional)** - `epsilon_decay`:
  - Decreases the epsilon value over time to gradually reduce exploration as the agent learns.

- **Minimum Epsilon** - `min_epsilon`:
  - Defines the lower bound for epsilon. Once epsilon decays to this value, it will no longer decrease further. This ensures the agent still explores occasionally even as it becomes more confident in its policy.
  - Typical range: [0.0, 1.0]
  - Ensures that exploration does not completely stop, which helps avoid getting stuck in local optima.

- **Number of Episodes** - `episodes`: 
  - The total number of episodes (complete runs) the agent will be trained on. More episodes help the agent to learn better.

- **Max Steps per Episode** - `max_steps`: 
  - The maximum number of steps the agent will take per episode before terminating it. This helps prevent infinite loops in environments that don't have a clear terminal state.

## Training Output Explanation
During the training process, you will see periodic updates that provide insights into the agent’s performance and its exploration strategy. 

Below is an example of the output:
```bash
is_slippery=False
reward_shaping=False
episode 0: total reward = 0.0, epsilon = 0.995
episode 100: total reward = 0.0, epsilon = 0.6027415843082742
episode 200: total reward = 1.0, epsilon = 0.36512303261753626
episode 300: total reward = 1.0, epsilon = 0.2211807388415433
episode 400: total reward = 1.0, epsilon = 0.13398475271138335
episode 500: total reward = 1.0, epsilon = 0.0811640021330769
episode 600: total reward = 1.0, epsilon = 0.04916675299948831
episode 700: total reward = 1.0, epsilon = 0.029783765425331846
episode 800: total reward = 1.0, epsilon = 0.018042124582040707
episode 900: total reward = 1.0, epsilon = 0.010929385683282892
episode 1000: total reward = 1.0, epsilon = 0.01
Training completed!
```

### What to Expect from the Output

- **Episode Number**: Each line represents the result after completing an episode (a complete run of the environment). In this example, it shows progress every 100 episodes for 1000 episodes total.
  
- **Total Reward**: The total reward accumulated during that episode. This is a measure of how well the agent performed, with higher values indicating better performance. Initially, the reward may be low as the agent explores the environment and learns, but over time, the reward should increase as the agent improves its policy. In the above output, the reward quickly reaches `1.0`, which is the optimal reward for this game.
    - **Reward**: The reward given by the environment at each step of the episode. The environment has a reward schedule that determines the points awarded for different actions:
        - **Reach goal (G)**: +1 (positive reward)
        - **Reach hole (H)**: 0 (no reward, since reward shaping is off)
        - **Reach frozen (F)**: 0 (no reward, since reward shaping is off)

- **Epsilon (ε)**: Epsilon represents the exploration rate of the agent. It controls the likelihood of choosing a random action (exploration) versus following the current best-known action (exploitation). 
  - High values of epsilon (close to 1) indicate that the agent is exploring more by choosing random actions.
  - Low values (close to 0) indicate that the agent is exploiting its learned policy and choosing the best-known actions.
  
  As you can see in the example, the epsilon value starts at `0.995` and gradually decays with each episode due to the `epsilon_decay` parameter. By the end of the training, epsilon stabilizes at `0.01`, meaning the agent has mostly switched to exploitation, relying on the learned policy.

### How to Interpret the Output

1. **Early Episodes (0-100)**:
   - In the beginning, the agent is exploring the environment and learning from random actions, which can result in low or no rewards.
   - Epsilon is still high, so exploration is prioritized over exploitation.

2. **Mid-Training (100-1000)**:
   - The agent starts gaining rewards as it learns the environment and its actions become more intentional.
   - Epsilon is decaying, which means the agent is starting to exploit its learned policy more while still exploring occasionally.

3. **Late Training (1000-10000)**:
   - The agent consistently earns the maximum reward (`1.0`), indicating that it has learned an effective strategy for interacting with the environment.
   - Epsilon has decreased to its minimum value (`0.01`), meaning the agent is now largely exploiting its knowledge and rarely choosing random actions.

## TODO
- [ ] `README.md`: Explain clearly what `is_slippery` and `reward_shaping` are.
- [ ] `test_main.py`: A test script.

## License

This project is licensed under the [MIT License](LICENSE).

You are free to use, modify, and distribute this code, as long as you include the original copyright notice and disclaimers in your project.