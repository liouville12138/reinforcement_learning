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
        x = x.squeeze(-1)
        return x


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
    ):
        self.memory_counter = 0
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.target_net = Net(n_features, n_actions)
        self.eval_net = Net(n_features, n_actions)

        self.loss_func = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))  # np.hstack将参数元组的元素数组按水平方向进行叠加
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, x):
        # to have batch dimension when feed into tf placeholder
        observation = torch.from_numpy(x[np.newaxis, :]).to(torch.float32)

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.eval_net(observation)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]  # 最大值索引作为动作
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('target_params_replaced\n')
        self.learn_step_counter += 1

        # sample batch memory from all memory
        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        state = torch.FloatTensor(batch_memory[:, :self.n_features])
        actions = torch.LongTensor(batch_memory[:, self.n_features:self.n_features + 1].astype(int))
        reward = torch.FloatTensor(batch_memory[:, self.n_features + 1:self.n_features + 2])
        state_ = torch.FloatTensor(batch_memory[:, -self.n_features:])

        q_eval = self.eval_net(state).gather(1, actions)         # shape (batch, 1)
        q_next = self.target_net(state_).detach()     # detach from graph, don't backpropagate
        q_target = reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_net(self, path):
        torch.save(self.target_net, path)

    def load_net(self, path):
        self.target_net = torch.load(path)
        self.eval_net = torch.load(path)
