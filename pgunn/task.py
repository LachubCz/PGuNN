"""
file contains class Task, which is wrapper
for working with different environments
"""
import os, inspect
import sys
import json
from keras import backend as K
import gym
import gym_2048
from agent import Agent
from visualization import combined_graph
from tools import agent_score_estimate, err_print

class Task:
    """
    class for working with different environments
    """
    def __init__(self, args):
        self.args = args
        self.name = None
        self.env = None
        self.env_state_size = None
        self.env_action_size = None
        self.type = None
        self.solved_score = None
        self.average_rand_score = None
        self.max_steps = None
        self.agent = None
        self.test = None
        self.load_json()


    def load_json(self):
        """
        method loads data about environment from json
        """
        self.name = self.args.environment
        self.env = gym.make(self.name)

        with open(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/task_args.json') as f:
            data = json.load(f)
        self.type = data[self.args.environment]["type"]

        if self.type == "image":
            self.env_state_size = (self.args.frames, 84, 84)
        else:
            self.env_state_size = self.env.observation_space.shape[0]
        self.env_action_size = self.env.action_space.n

        if data[self.args.environment]["solved_score"] != "None":
            self.solved_score = int(data[self.args.environment]["solved_score"])

        if data[self.args.environment]["average_rand_score"] != "None":
            self.average_rand_score = float(data[self.args.environment]["average_rand_score"])

        if data[self.args.environment]["max_steps"] != "None":
            self.max_steps = int(data[self.args.environment]["max_steps"])

        self.agent = Agent(self.env_state_size, self.env_action_size, self.args)

        if self.name in set(["CartPole-v0", "CartPole-v1", "MountainCar-v0"]):
            self.test = self.ohe_test
        elif self.name == "Acrobot-v1":
            self.test = self.aohe_test
        else:
            self.test = None


    def ohe_test(self, scores, episode_numbers):
        """
        method runs 100 testing episodes
        """
        complete_estimation = agent_score_estimate(self, 10)
        if complete_estimation >= self.solved_score:
            for i in range(2, 11):
                estimation = agent_score_estimate(self, 10)
                complete_estimation = complete_estimation + estimation
                if (complete_estimation / i) < self.solved_score:
                    return
        else:
            return
        score = complete_estimation / 10

        if score > self.solved_score:
            if not self.args.dont_save:
                self.agent.save_model_weights("{}-solved.h5" .format(self.name))
                combined_graph(scores, episode_numbers, "{}_results" .format(self.name),
                               [episode_numbers[-1], max(scores)+10], {self.average_rand_score:self.average_rand_score}, scatter=True)
            if self.solved_score <= score:
                print("[Task was solved after {} episodes with score {}.]" .format(episode_numbers[-1], score))
            else:
                print("[Task wasn't solved after {} episodes with score {}.]" .format(episode_numbers[-1], score))
            print("[SUCCESSFUL RUN]")
            K.clear_session()
            sys.exit()


    def aohe_test(self, scores, episode_numbers):
        """
        method runs 100 testing episodes after 100 training episodes
        """
        if episode_numbers[-1] == 99:
            score = agent_score_estimate(self, 100)
            if not self.args.dont_save:
                self.agent.save_model_weights("{}-solved.h5" .format(self.name))
                combined_graph(scores, episode_numbers, "{}_results" .format(self.name), 
                               [episode_numbers[-1], max(scores)+10], {self.average_rand_score:self.average_rand_score}, scatter=True)
            if self.solved_score <= score:
                print("[Task was solved after {} episodes with score {}.]" .format(episode_numbers[-1], score))
            else:
                print("[Task wasn't solved after {} episodes with score {}.]" .format(episode_numbers[-1], score))
            print("[SUCCESSFUL RUN]")
            K.clear_session()
            sys.exit()
