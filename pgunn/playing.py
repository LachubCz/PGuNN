"""
file contains class for testing of agent's game skills and saving memories without training
"""
import scipy
import numpy as np

class Playing():
    """
    class contains methods for testing of agent's game skills and methods for saving memories without training
    """
    def rand_score_estimate(task, games):
        """
        random agent score estimation 
        """
        total_reward = 0

        for game in range(games):
            done = False

            while not done:
                action = np.random.randint(0, task.env_action_size, size=1)[0]
                next_state, reward, done, info = task.env.step(action)

                total_reward = total_reward + reward

        return total_reward / games

    def score_estimate_vect(task, games):
        """
        agent score estimation
        """
        total_reward = 0

        for game in range(games):
            state = task.env.reset()
            done = False

            while not done:

                action = task.agent.get_action(state, epsilon=False)
                next_state, reward, done, info = task.env.step(action)

                state = next_state
                total_reward = total_reward + reward

        return total_reward / games

    def prior_rand_agent_replay_vect(task, episodes, observetime):
        """
        saving memories from random agent to prioritized memory
        """
        new_observation = 0
        task.agent.clear_memory()

        for eps in range(episodes):
            state = task.env.reset()

            for t in range(observetime):
                action = np.random.randint(0, task.env_action_size, size=1)[0]
                next_state, reward, done, info = task.env.step(action)

                task.agent.remember(state, action, reward, next_state, done, rand_agent=True)
                new_observation = new_observation + 1
                state = next_state

                if task.agent.memory.length == task.agent.memory_size:
                    return new_observation

                if done:
                    break

    def prior_agent_replay_vect(task, episodes, observetime):
        """
        saving memories from agent to prioritized memory
        """
        new_observation = 0
        task.agent.clear_memory()

        for eps in range(episodes):
            state = task.env.reset()

            for t in range(observetime):
                action = task.agent.get_action(state, epsilon=True)
                next_state, reward, done, info = task.env.step(action)

                task.agent.remember(state, action, reward, next_state, done, rand_agent=True)
                new_observation = new_observation + 1
                state = next_state

                if task.agent.memory.length == task.agent.memory_size:
                    return new_observation

                if done:
                    break

        return new_observation

    def rand_agent_replay_vect(task, episodes, observetime):
        """
        saving memories from random agent to basic memory
        """
        new_observation = 0
        task.agent.clear_memory()

        for eps in range(episodes):
            state = task.env.reset()

            for t in range(observetime):
                action = np.random.randint(0, task.env_action_size, size=1)[0]
                next_state, reward, done, info = task.env.step(action)

                task.agent.remember(state, action, reward, next_state, done, rand_agent=True)
                new_observation = new_observation + 1
                state = next_state
                
                if len(task.agent.memory) == task.agent.memory_size:
                    return new_observation

                if done:
                    break

    def agent_replay_vect(task, episodes, observetime):
        """
        saving memories from agent to basic memory
        """
        new_observation = 0
        task.agent.clear_memory()

        for eps in range(episodes):
            state = task.env.reset()

            for t in range(observetime):
                action = task.agent.get_action(state, epsilon=True)
                next_state, reward, done, info = task.env.step(action)

                task.agent.remember(state, action, reward, next_state, done, rand_agent=False)
                new_observation = new_observation + 1
                state = next_state
                
                if len(task.agent.memory) == task.agent.memory_size:
                    return new_observation

                if done:
                    break
