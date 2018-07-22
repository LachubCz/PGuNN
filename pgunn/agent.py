"""
file contains implementation of Agent class
"""
import random
import numpy as np
from collections import deque
from memory import Memory
from network import Network
from tools import split_2048, possible_moves

class Agent:
    """
    class implements agent
    """
    def __init__(self, algorithm, state_size, action_size, model_type, memory_type, args, net_units=None):
        self.args = args
        self.initial_epsilon = 1
        self.final_epsilon = 0.1
        self.current_epsilon = self.initial_epsilon
        self.epsilon_decay = 0.0000009
        self.gamma = 0.99
        self.minibatch_size = 128
        self.learning_rate = 0.0005
        self.fraction_update = 0.125

        self.memory_type = memory_type
        self.memory_size = 2500
        if self.memory_type == "basic":
            self.memory = deque(maxlen=self.memory_size)
        else:
            self.memory = Memory(self.memory_size)

        self.action_size = action_size
        self.state_size = state_size

        network = Network(state_size, action_size, self.learning_rate, "MSE", [True, False])

        self.model_type = model_type
        if model_type == "2layer_bsc_mdl":
            self.model_net = network.make_2layer_mdl(net_units)
            self.target_net = network.make_2layer_mdl(net_units)
        elif model_type == "2layer_duel_mdl":
            self.model_net = network.make_2layer_duel_mdl(net_units)
            self.target_net = network.make_2layer_duel_mdl(net_units)
        elif model_type == "bsc_img_mdl":
            self.model_net = network.make_bsc_img_mdl()
            self.target_net = network.make_bsc_img_mdl()
        elif model_type == "duel_img_model":
            self.model_net = network.make_duel_img_mdl()
            self.target_net = network.make_duel_img_mdl()
        elif model_type == "1layer_ram_mdl":
            self.model_net = network.make_1layer_mdl(net_units)
            self.target_net = network.make_1layer_mdl(net_units)
        elif model_type == "experimental":
            self.model_net = network.make_4layer_mdl([128, 128, 128, 128])
            self.target_net = network.make_4layer_mdl([128, 128, 128, 128])

        self.update_target_net()

        self.algorithm = algorithm
        self.algorithms = {"DQN" : self.train_dqn,
                           "DQN+TN" : self.train_target_dqn,
                           "DDQN" : self.train_ddqn,
                          }

    def update_target_net(self):
        """
        method updates target network
        """
        self.target_net.set_weights(self.model_net.get_weights())
        print("[Target network was updated.]")

    def update_target_net_partially(self):
        """
        method updates target network by parts
        """
        weights_model = self.model_net.get_weights()
        weights_target = self.target_net.get_weights()

        for i in range(len(weights_target)):
            weights_target[i] = weights_model[i] * self.fraction_update + weights_target[i] * (1 - self.fraction_update)

        self.target_net.set_weights(weights_target)
        print("[Target network was updated by parts.]")

    def get_error(self, state, action, reward, next_state, done):
        """
        method returns difference between Q-value from primary and target network
        """
        q_value = self.model_net.predict(np.array([state]))
        ns_model_pred = self.model_net.predict(np.array([next_state]))
        ns_target_pred = self.target_net.predict(np.array([next_state]))

        obs_error = q_value[0][action]

        if done == 1:
            q_value[0][action] = reward
        else:
            q_value[0][action] = reward + self.gamma * ns_target_pred[0][np.argmax(ns_model_pred)]

        obs_error = abs(obs_error - q_value[0][action])

        return obs_error

    def remember(self, state, action, reward, next_state, done, rand_agent):
        """
        method saves observation (experience) to experience replay memory
        """
        if self.memory_type == "basic":
            self.memory.append((state, action, reward, next_state, done))
        else:
            if rand_agent:
                obs_error = abs(reward)
            else:
                obs_error = self.get_error(state, action, reward, next_state, done)

            self.memory.add_observation((state, action, reward, next_state, done), obs_error)

    def clear_memory(self):
        """
        method clears replay memory
        """
        self.memory.clear()

    def decrease_epsilon(self):
        """
        method decreases epsilon
        """
        if self.current_epsilon > self.final_epsilon:
            if (self.current_epsilon - self.epsilon_decay) > self.final_epsilon:
                self.current_epsilon = self.current_epsilon - self.epsilon_decay
            else:
                self.current_epsilon = self.final_epsilon

    def get_action(self, task, state, non_normalized_state, epsilon):
        """
        method returns action to take
        """
        if not epsilon:
            q_value = self.model_net.predict(np.array([state]))
        else:
            if np.random.rand() <= self.current_epsilon:
                if task.name == "2048-v0":
                    possible_actions = possible_moves(non_normalized_state)
                    while True:
                        rand_action = np.random.randint(0, self.action_size, size=1)[0]
                        if possible_actions[rand_action] == 1:
                            return rand_action
                else:
                    return np.random.randint(0, self.action_size, size=1)[0]
            else:
                q_value = self.model_net.predict(np.array([state]))

        if task.name == "2048-v0":
            possible_actions = possible_moves(non_normalized_state)
            while True:
                chosen_action = np.argmax(q_value)
                if possible_actions[chosen_action] == 1:
                    return chosen_action
                else:
                    q_value[0][chosen_action] = -100

        return np.argmax(q_value)

    def get_minibatch(self):
        """
        method returns minibatch from diffrent memory types
        """
        if self.memory_type == "basic":
            minibatch = random.sample(list(self.memory), self.minibatch_size)
            state = np.array([i[0] for i in minibatch])
            action = [i[1] for i in minibatch]
            reward = [i[2] for i in minibatch]
            next_state = np.array([i[3] for i in minibatch])
            done = [i[4] for i in minibatch]
        else:
            minibatch = self.memory.sample(self.minibatch_size)

            state = np.array([i[1][0] for i in minibatch])
            action = [i[1][1] for i in minibatch]
            reward = [i[1][2] for i in minibatch]
            next_state = np.array([i[1][3] for i in minibatch])
            done = [i[1][4] for i in minibatch]

        return state, action, reward, next_state, done

    def train(self):
        """
        method trains agent with selected algorithm
        """
        self.algorithms[self.algorithm]()

    def train_dqn(self):
        """
        method trains agent using DQN
        """
        #print(self.memory)
        if self.memory_type == "basic":
            if len(self.memory) >= self.minibatch_size:
                state, action, reward, next_state, done = self.get_minibatch()
            else:
                return
        else:
            if self.memory.length >= self.minibatch_size:
                state, action, reward, next_state, done = self.get_minibatch()
            else:
                return

        errors = np.zeros(self.minibatch_size)
        """
        if self.model_type == "experimental":
            state = split_2048(state)
            next_state = split_2048(next_state)
            q_value = self.model_net.predict(state)
            ns_model_pred = self.model_net.predict(next_state)
        else:"""
        possible_actions_curr = []
        if self.args.environment == "2048-v0": 
            for i, item in enumerate(state):
                possible_actions_curr.append(possible_moves(item))

            state = state / 16384.0 - 0.5
            next_state = next_state / 16384.0 - 0.5

        q_value = self.model_net.predict(np.array(state))
        ns_model_pred = self.model_net.predict(np.array(next_state))

        for i in range(0, self.minibatch_size):
            errors[i] = q_value[i][action[i]]

            if done[i] == 1:
                q_value[i][action[i]] = reward[i]
            else:
                q_value[i][action[i]] = reward[i] + self.gamma * np.max(ns_model_pred[i])

            errors[i] = abs(errors[i] - q_value[i][action[i]])

        for i, item in enumerate(possible_actions_curr):
            for e, elem in enumerate(item):
                if elem == 0:
                    q_value[i][e] = -1

        self.model_net.fit(state, q_value, epochs=1, verbose=0)
        if self.memory_type == "dueling":
            self.memory.update_minibatch(minibatch, errors)

    def train_target_dqn(self):
        """
        method trains agent using DQN with target network
        """
        if self.memory_type == "basic":
            if len(self.memory) >= self.minibatch_size:
                state, action, reward, next_state, done = self.get_minibatch()
            else:
                return
        else:
            if self.memory.length >= self.minibatch_size:
                state, action, reward, next_state, done = self.get_minibatch()
            else:
                return

        errors = np.zeros(self.minibatch_size)
        """
        if self.model_type == "experimental":
            state = split_2048(state)
            next_state = split_2048(next_state)
            q_value = self.model_net.predict(state)
            ns_model_pred = self.model_net.predict(next_state)
        else:"""
        possible_actions_curr = []
        if self.args.environment == "2048-v0": 
            for i, item in enumerate(state):
                possible_actions_curr.append(possible_moves(item))

            state = state / 16384.0 - 0.5
            next_state = next_state / 16384.0 - 0.5

        q_value = self.model_net.predict(np.array(state))
        ns_model_pred = self.model_net.predict(np.array(next_state))

        for i in range(0, self.minibatch_size):
            errors[i] = q_value[i][action[i]]

            if done[i] == 1:
                q_value[i][action[i]] = reward[i]
            else:
                q_value[i][action[i]] = reward[i] + self.gamma * np.max(ns_target_pred[i])

            errors[i] = abs(errors[i] - q_value[i][action[i]])

        for i, item in enumerate(possible_actions_curr):
            for e, elem in enumerate(item):
                if elem == 0:
                    q_value[i][e] = -1

        self.model_net.fit(state, q_value, epochs=1, verbose=0)
        if self.memory_type == "dueling":
            self.memory.update_minibatch(minibatch, errors)

    def train_ddqn(self):
        """
        method trains agent using DDQN
        """
        if self.memory_type == "basic":
            if len(self.memory) >= self.minibatch_size:
                state, action, reward, next_state, done = self.get_minibatch()
            else:
                return
        else:
            if self.memory.length >= self.minibatch_size:
                state, action, reward, next_state, done = self.get_minibatch()
            else:
                return

        errors = np.zeros(self.minibatch_size)
        """
        if self.model_type == "experimental":
            state = split_2048(state)
            next_state = split_2048(next_state)
            q_value = self.model_net.predict(state)
            ns_model_pred = self.model_net.predict(next_state)
            ns_target_pred = self.target_net.predict(next_state)
        else:"""
        possible_actions_curr = []
        if self.args.environment == "2048-v0": 
            for i, item in enumerate(state):
                possible_actions_curr.append(possible_moves(item))

            state = state / 16384.0 - 0.5
            next_state = next_state / 16384.0 - 0.5

        q_value = self.model_net.predict(state)
        ns_model_pred = self.model_net.predict(next_state)
        ns_target_pred = self.target_net.predict(next_state)

        for i in range(0, self.minibatch_size):
            errors[i] = q_value[i][action[i]]

            if done[i] == 1:
                q_value[i][action[i]] = reward[i]
            else:
                q_value[i][action[i]] = reward[i] + self.gamma * ns_target_pred[i][np.argmax(ns_model_pred[i])]

            errors[i] = abs(errors[i] - q_value[i][action[i]])

        for i, item in enumerate(possible_actions_curr):
            for e, elem in enumerate(item):
                if elem == 0:
                    q_value[i][e] = -1

        self.model_net.fit(state, q_value, epochs=1, verbose=0)
        if self.memory_type == "dueling":
            self.memory.update_minibatch(minibatch, errors)

    def load_model_weights(self, name):
        """
        method loads weights to primary neural network
        """
        self.model_net.load_weights(name)
        print("[Model has been loaded from \"{}\".]" .format(name))

    def save_model_weights(self, name):
        """
        method saves weights of primary neural network
        """
        self.model_net.save_weights("./model-{}".format(name))
        print("[Model was saved to \"./model-{}\".]" .format(name))

    def load_target_weights(self, name):
        """
        method loads weights to target neural network
        """
        self.target_net.load_weights(name)
        print("[Target model has been loaded from \"{}\".]" .format(name))

    def save_target_weights(self, name):
        """
        method saves weights of target neural network
        """
        self.target_net.save_weights("./target-{}".format(name))
        print("[Target model was saved to \"./target-{}\".]" .format(name))
