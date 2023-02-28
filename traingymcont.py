from stable_baselines3.common.env_checker import check_env
from environment import ARPOD_GYM
from dynamics import chaser_continous
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable
import gym
import torch as T
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
#from stable_baselines3.common.schedules import LinearSchedule
"""
def make_env(rank: int, seed: int = 0) -> Callable:
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    def _init() -> gym.Env:
        chaser = chaser_continous(True, False)
        env = ARPOD_GYM(chaser, True)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


num_cpu = 4  # Number of processes to use
# Create the vectorized environment
env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
#env = ARPOD_GYM(chaser, True)
# It will check your custom environment and output additional warnings if needed
#check_env(env)
"""


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

T.cuda.empty_cache()
T.autocast(device_type='cuda')
T.backends.cudnn.benchmark = True
T.backends.cudnn.enabled = True
#T.cuda.set_per_process_memory_fraction(0.8, device='cuda:0')
#chaser = chaser_continous(True, False)
#env = ARPOD_GYM(chaser, True)
env = make_vec_env(ARPOD_GYM, n_envs=12)
env = VecNormalize(env, norm_obs=True, norm_reward=True)
episodes = 200000.0
#policy_kwargs = dict(activation_fn=T.nn.LeakyReLU,
#                    net_arch=[dict(pi=[50, 50, 50], vf=[50, 50, 50])])

policy_kwargs = dict(activation_fn=T.nn.LeakyReLU, net_arch=[dict(pi=[130, 44, 30], vf=[145, 25, 5])], full_std=False, squash_output=False, log_std_init=-2.0, ortho_init=False)

training_steps = 1500.0 * episodes
model = PPO("MlpPolicy", env,  gamma=0.99, clip_range=0.4, learning_rate = linear_schedule(0.0005), n_steps=750, ent_coef=0.01, target_kl=0.001, batch_size=1500, 
             policy_kwargs=policy_kwargs, verbose=1, device='cuda', gae_lambda=0.95, n_epochs=20, normalize_advantage=True, max_grad_norm=0.5, use_sde=True, sde_sample_freq=4, tensorboard_log="./ppo_arpodcont_tensorboard/")
model.learn(total_timesteps=training_steps, progress_bar=True, tb_log_name="ppocontlogs")
model.save("ppo_arpodcontvbarbehind")
