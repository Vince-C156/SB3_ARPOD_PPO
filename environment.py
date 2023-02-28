from reward_shaping import reward_formulation
from visualizer_close import write2text
#write2text(chaser, data_dir, file_name, step)
import numpy as np
#import cupy as np
import pdb
import gym
from gym import spaces
import os
from dynamics import chaser_continous

class ARPOD:

    def __init__(self, chaser):
        self.r_form = reward_formulation(chaser)
        self.chaser = chaser
        self.info = {'time in los' : 0,
                     'time in validslowzone' : 0,
                     'episode time' : 0,
                     'time in los and slowzone' : 0,
                     'time in los and phase3' : 0}

    def step(self, action):
        reward = 0
        done = False
        obs = self.chaser.get_next(action)
        self.chaser.update_state(obs)

        p, done = self.r_form.terminal_conditions()
        reward += p

        if done:
            return obs, reward, done, self.info
        """
        r, done = self.r_form.win_conditions()
        reward += r
        """
        if done:
            return obs, reward, done, self.info

        p = self.r_form.soft_penalities()
        reward += p

        r = self.r_form.soft_rewards()
        reward += r
        self.info['time in los'] = self.info['time in los'] + self.r_form.time_inlos
        #self.info['time in validslowzone'] = self.info['time in validslowzone'] + self.r_form.time_slowzone
        #self.info['time in los and slowzone'] = self.info['time in los and slowzone'] + self.r_form.time_inlos_slowzone
        #self.info['time in los and phase 3'] = self.info['time in los and phase3'] + self.r_form.time_inlos_phase3
        self.info['episode time'] = self.info['episode time'] + 1
        self.r_form.reset_counts()

        #i = len(os.listdir('runs/vbar1'))
        #print(f'write index {i}')
        #data_file_name = f'vbar1/chaser{i}.txt'
        #self.write_data(data_file_name)
        return obs, reward, done, self.info

    def write_data(self, file_name):
        #print('writing data to text')
        #print(f'current step {self.chaser.current_step}')
        write2text(self.chaser, 'runs', file_name, self.chaser.current_step)


    def reset(self):
        self.chaser.reset()
        self.chaser.update_state(self.chaser.state)
        self.info = {'time in los' : 0,
                     'time in validslowzone' : 0,
                     'episode time' : 0,
                     'time in los and slowzone' : 0,
                     'time in los and phase3' : 0}

        print('reseting environment')
        return np.array(self.chaser.state, copy=True)


class ARPOD_GYM(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        #self.num_envs = 1
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-10.0, high=10.0,
                                            shape=(3,), dtype=np.float64)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-5000.0, high=5000.0,
                                            shape=(6,), dtype=np.float64)

        self.episode = 0
        #self.r_form = reward_formulation(chaser)
        self.chaser = chaser_continous(True, False)
        self.r_form = reward_formulation(self.chaser)
        self.info = {'time in los' : 0,
                     'time in validslowzone' : 0,
                     'episode time' : 0,
                     'time in los and slowzone' : 0,
                     'time in los and phase3' : 0}
        self.iscontinous = True
        model_id = len(os.listdir('runs/'))
        self.runs_dir = f'vbar{model_id}'

        print(f'creating runs log in directory {self.runs_dir}')
        os.mkdir(f'runs/{self.runs_dir}')

    def step(self, action):
        reward = 0
        done = False
        action = np.clip(action, -10.0, 10.0, dtype=np.float64)
        #action *= 10.0
        obs = self.chaser.get_next(action)
        self.chaser.update_state(obs)

        p, done = self.r_form.terminal_conditions()
        reward += p

        if done:
            return obs, reward, done, self.info
        """
        r, done = self.r_form.win_conditions()
        reward += r
        """
        if done:
            return obs, reward, done, self.info

        p = self.r_form.soft_penalities()
        reward += p

        r = self.r_form.soft_rewards()
        reward += r
        self.info['time in los'] = self.info['time in los'] + self.r_form.time_inlos
        #self.info['time in validslowzone'] = self.info['time in validslowzone'] + self.r_form.time_slowzone
        #self.info['time in los and slowzone'] = self.info['time in los and slowzone'] + self.r_form.time_inlos_slowzone
        #self.info['time in los and phase 3'] = self.info['time in los and phase3'] + self.r_form.time_inlos_phase3
        self.info['episode time'] = self.info['episode time'] + 1
        self.r_form.reset_counts()
        data_file_name = f'{self.runs_dir}/chaser{self.episode}.txt'
        self.write_data(data_file_name)

        return obs, reward, done, self.info

    def reset(self):
        print('resetting and showing info')
        print(self.info)
        self.chaser.reset()
        self.chaser.update_state(self.chaser.state)
        self.info = {'time in los' : 0,
                     'time in validslowzone' : 0,
                     'episode time' : 0,
                     'time in los and slowzone' : 0,
                     'time in los and phase3' : 0}

        print('reseting environment')
        observation = np.array(self.chaser.state, copy=True)
        self.episode += 1
        return observation

    def render(self, mode="human"):
        pass

    def write_data(self, file_name):
        #print('writing data to text')
        #print(f'current step {self.chaser.current_step}')
        write2text(self.chaser, 'runs', file_name, self.chaser.current_step)
