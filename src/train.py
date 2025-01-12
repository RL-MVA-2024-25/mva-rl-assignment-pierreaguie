from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import torch.nn as nn
import torch.optim as optim
import torch

import numpy as np
import random
import matplotlib.pyplot as plt

from copy import deepcopy
import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


default_args = {
    "hidden_dim": 500,
    "n_hidden_layers": 5, # 5,

    "lr": 1e-3,
    "batch_size": 128,

    "nb_gradient_steps": 5,
    "update_target_freq": 400,

    "capacity": 10000,

    "gamma": .99,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_stop": 20000,
    "epsilon_delay": 100,
}


class ProjectAgent:

    def __init__(self, args = default_args):

        self.env = env
        self.n_episodes_steps = self.env._max_episode_steps
        self.n_episodes = 500
        
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.n_hidden_layers = args["n_hidden_layers"]
        self.hidden_dim = args["hidden_dim"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu"))
        self.model = self.get_model(state_dim = self.state_dim, n_actions = self.n_actions, n_hidden_layers = self.n_hidden_layers, hidden_dim = self.hidden_dim).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.target_model.eval()

        self.lr = args["lr"]
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.batch_size = args["batch_size"]

        self.nb_gradient_steps = args["nb_gradient_steps"]
        self.update_target_freq = args["update_target_freq"]

        self.capacity = args["capacity"]
        self.memory = ReplayBuffer(self.capacity, self.device)

        self.gamma = args["gamma"]
        self.epsilon_max = args["epsilon_max"]
        self.epsilon_min = args["epsilon_min"]
        self.epsilon_stop = args["epsilon_stop"]
        self.epsilon_delay = args["epsilon_delay"]
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop


    def act(self, observation, use_random=False):
        if use_random:
            return self.env.action_space.sample()
        else:
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
            q_values = self.model(observation)
            return torch.argmax(q_values).item()


    def save(self, path):
        torch.save(self.model.state_dict(), path)


    def load(self):
        self.model.load_state_dict(torch.load("models/best_model.pth", map_location=self.device, weights_only=True))


    def get_model(self, state_dim, n_actions, n_hidden_layers, hidden_dim):
        model = nn.Sequential()
        model.add_module("input", nn.Linear(state_dim, hidden_dim))
        model.add_module("relu0", nn.ReLU())
        for i in range(n_hidden_layers):
            model.add_module(f"hidden_{i}", nn.Linear(hidden_dim, hidden_dim))
            model.add_module(f"relu_{i}", nn.ReLU())
        model.add_module("output", nn.Linear(hidden_dim, n_actions))
        return model
    

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            next_actions = self.model(Y).argmax(1).unsqueeze(1)
            with torch.no_grad():
                QYmax = self.target_model(Y).gather(1, next_actions).squeeze(1).detach()
            update = R + self.gamma * QYmax * (1 - D)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))

            loss = self.loss_fn(QXA, update.unsqueeze(1))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()


    def normalize_state(self, state):
        return (state - self.state_mean)/self.state_std
        

    def train(self):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = self.env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_reward = -np.inf

        while episode < self.n_episodes:
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

            use_random = np.random.rand() < epsilon
            action = self.act(state, use_random)

            next_state, reward, done, trunc, _ = self.env.step(action)
            self.memory.push(state, action, reward, next_state, done)
            episode_cum_reward += reward

            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            if step % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            step += 1
            if done or trunc:
                episode_return.append(episode_cum_reward)
                episode += 1
                state, _ = self.env.reset()
                episode_cum_reward = 0
                print(f"Episode {episode} - Reward: {episode_return[-1]:.3f}, epsilon: {epsilon}")

                if episode_return[-1] > best_reward:
                    print(f"New best reward: {episode_return[-1]}")
                    best_reward = episode_return[-1]
                    self.save("models/model.pth")
            else:
                state = next_state

        return episode_return



class ReplayBuffer:

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    
    def __len__(self):
        return len(self.buffer)
    



if __name__ == "__main__":
    agent = ProjectAgent()
    print(agent.device)
    rewards = agent.train()
    plt.plot(rewards)
    plt.show()