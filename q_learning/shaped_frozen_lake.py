import gym


class ShapedFrozenLake(gym.Env):
    """
    A custom environment that modifies the reward structure of the FrozenLake environment in OpenAI Gym.
    This environment applies reward shaping to provide additional rewards when the agent approaches the goal.

    Parameters:
    - is_slippery (bool): If set to True, the environment will have slippery tiles, making movement less deterministic. If set to False, the environment is deterministic.
    - reward_shaping (bool): If set to True, additional rewards or penalties are provided to the agent based on proximity to the goal.

    Reward Structure:
    - Standard FrozenLake environment provides a reward of 1 for reaching the goal and 0 otherwise.
    - ShapedFrozenLake can give additional intermediate rewards or penalties for steps that bring the agent closer to the goal.

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
    distance_to_goal = abs(
        current_position[0] - goal_position[0]) + abs(current_position[1] - goal_position[1])

    # if the agent fell into a hole, penalize based on distance to goal
    if reward == 0:
        return -0.05 * distance_to_goal

    # if the agent reached the goal, no further shaping (reward is already 1)
    return reward
