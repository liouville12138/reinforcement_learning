import torch
import numpy as np

import actor
import critic


class DDPG:
    def __init__(self, memory_size, action_feature, state_feature, learning_rate, batch_size, gamma, tau):
        self.memory = np.zeros((memory_size, state_feature * 2 + action_feature + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_size = memory_size
        self.action_feature = action_feature
        self.state_feature = state_feature
        self.target_update = tau

        self.actor = actor.Actor(action_feature, state_feature, learning_rate)
        self.critic = critic.Critic(action_feature, state_feature, learning_rate, gamma)
        self.batch_size = batch_size

    def store_transition(self, state, action, reward, state_):
        transition = np.hstack((state, action.ravel(), [reward], state_))
        index = self.pointer % self.memory_size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def choose_action(self, s):
        observation = torch.from_numpy(s[np.newaxis, :]).to(torch.float32)
        return self.actor.choose_action(observation)

    def _soft_update(self, net_target, net, tau):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def learn(self):
        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        memory = self.memory[sample_index, :]
        state = torch.FloatTensor(memory[:, :self.state_feature])
        action = torch.FloatTensor(memory[:, self.state_feature:self.state_feature + self.action_feature])
        reward = torch.FloatTensor(memory[:, -self.state_feature - 1:-self.state_feature])
        state_ = torch.FloatTensor(memory[:, -self.state_feature:])

        action_ = self.actor.target_net(state_).detach()
        self.critic.learn(state, action, reward, state_, action_)

        actor_eval = self.actor.eval_net(state)
        critic_eval = self.critic.eval_net(torch.cat((state, actor_eval), 1))
        self.actor.learn(critic_eval)

        self._soft_update(self.actor.target_net, self.actor.eval_net, self.target_update)
        self._soft_update(self.critic.target_net, self.critic.eval_net, self.target_update)

    def save_net(self, path):
        torch.save({
            "actor": self.actor.target_net,
            "critic": self.critic.target_net,
        })

    def load_net(self, path):
        checkpoint = torch.load(path)
        self.actor.target_net = checkpoint["actor"]
        self.actor.eval_net = checkpoint["actor"]
        self.critic.target_net = checkpoint["critic"]
        self.critic.eval_net = checkpoint["critic"]

