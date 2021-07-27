"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import torch
import numpy as np
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, input_size, output_num):
        super(Net, self).__init__()
        self.full_connect = torch.nn.Sequential(  # (batch_size , 3, input_size)
            torch.nn.Linear(input_size, 50),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(50, output_num),
        )
        self.full_connect[0].weight.data.normal_(0, 0.1)  # initialization
        self.full_connect[2].weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.full_connect.forward(x)
        x = F.softmax(x, 1)
        x = x.squeeze(-1)
        return x


# Deep Q Network off-policy
class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            gamma=0.9,
            batch_size=32,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = gamma
        self.batch_size = batch_size

        self.state, self.action, self.reward = [], [], []

        self.net = Net(n_features, n_actions)
        self.loss_func = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def store_transition(self, s, a, r):  # 整局更新，因此不存在memory的说法
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)

    def choose_action(self, x):
        # to have batch dimension when feed into tf placeholder
        #  不需要epsilon-greedy，因为概率本身就具有随机性
        observation = torch.from_numpy(x[np.newaxis, :]).to(torch.float32)
        prob_weights = self.net(observation).detach().numpy()
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_reward = np.zeros_like(self.reward)
        running_add = 0
        for t in reversed(range(0, len(self.reward))):
            running_add = running_add * self.gamma + self.reward[t]
            discounted_reward[t] = running_add

        # normalize episode rewards
        discounted_reward -= np.mean(discounted_reward)
        discounted_reward /= np.std(discounted_reward)
        return torch.FloatTensor(discounted_reward)

    def learn(self):
        # discount and normalize episode reward
        discounted_reward_norm = self._discount_and_norm_rewards()

        state = torch.FloatTensor(np.array(self.state))

        prob_weights = self.net(state)
        log_prob = torch.log(prob_weights)

        one_hot_action = F.one_hot(torch.LongTensor(np.array(self.action)), self.n_actions)

        action_log_prob = torch.max(-log_prob * one_hot_action, 1)[0]

        loss = torch.mean(action_log_prob * discounted_reward_norm)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # empty episode data
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        return discounted_reward_norm

    def save_net(self, path):
        torch.save(self.net, path)

    def load_net(self, path):
        self.net = torch.load(path)
