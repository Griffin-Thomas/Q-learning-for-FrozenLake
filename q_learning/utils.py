import argparse


def str2bool(value: str) -> bool:
    """
    Convert a string to a boolean value. 
    Accepts:
        'yes', 'true', 't', '1' -> True
        'no', 'false', 'f', '0' -> False

    Parameters:
    - value: string value passed from command-line arguments

    Returns:
    - bool: Converted boolean value
    """
    value = value.lower()

    if value in ['yes', 'y', 'true', 't', '1']:
        return True
    elif value in ['no', 'n', 'false', 'f', '0']:
        return False

    # otherwise
    raise argparse.ArgumentTypeError(
        "Value must be one of 'yes', 'no', 'y', 'n', 'true', 'false', 't', 'f', '1', or '0'."
    )


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

    # parameters for slippery and reward shaping
    # uses str2bool to allow multiple boolean representations
    parser.add_argument('--is_slippery', type=str2bool,
                        default=False, help="If True, environment is slippery.")
    parser.add_argument('--reward_shaping', type=str2bool,
                        default=False, help="If True, reward shaping is applied.")

    # hyperparameters
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='learning rate (alpha)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor (gamma)')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='exploration rate (epsilon)')
    parser.add_argument('--epsilon_decay', type=float,
                        default=0.995, help='decay rate for epsilon')
    parser.add_argument('--min_epsilon', type=float,
                        default=0.01, help='minimum value of epsilon')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='number of training episodes')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='maximum steps per episode')

    return parser.parse_args()
