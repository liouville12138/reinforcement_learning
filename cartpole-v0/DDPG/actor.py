import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, input_size, output_num):
        super(Net, self).__init__()
        self.full_connect = torch.nn.Sequential(  # (batch_size , 3, input_size)
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, output_num),
        )
        self.full_connect[0].weight.data.normal_(0, 0.1)  # initialization
        self.full_connect[2].weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.full_connect.forward(x)
        x = F.softmax(x, 1)
        x = x.squeeze(-1)
        return x


class Actor:
    def __init__(self, action_feature, state_feature, learning_rate):
        self.feature_num = action_feature
        self.lr = learning_rate

        self.eval_net = Net(state_feature, action_feature)
        self.target_net = Net(state_feature, action_feature)
        self.loss_func = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)

    def learn(self, critic_eval):
        loss = -torch.mean(critic_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def choose_action(self, state):
        return self.target_net(state).detach().numpy()
