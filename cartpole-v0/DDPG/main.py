import os
import gym
import ddpg
import numpy as np

def update(module, env, file_saved):
    for episode in range(10000):
        # initial observation
        observation = env.reset()
        episode_reward = 0
        step = 0
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action_prob = module.choose_action(observation)
            # add randomness to action selection for exploration

            # RL take action and get next observation and reward

            action = np.random.choice(range(action_prob.shape[1]),
                                      p=action_prob.ravel())  # select action w.r.t the actions prob

            observation_, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2


            module.store_transition(observation, action_prob, r, observation_)

            episode_reward += reward

            if module.pointer > module.memory_size:
                module.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            step += 1
            if done:
                print("Episode {} finished after {} timesteps".format(episode, step))
                break

    # end of game
    module.save_net(file_saved)
    print('game over')


if __name__ == "__main__":
    gymenv = gym.make('CartPole-v0')
    gymenv = gymenv.unwrapped  # 放开步数限制

    RL = ddpg.DDPG(memory_size=100,
                   action_feature=gymenv.action_space.n,
                   state_feature=gymenv.observation_space.shape[0],
                   learning_rate=0.001,
                   batch_size=32,
                   gamma=0.99,
                   tau=0.02)
    ddpg_net_file = "dqn_net.pickle"
    actions = list(range(gymenv.action_space.n))

    if os.path.exists(ddpg_net_file):
        RL.load_net(ddpg_net_file)
    update(RL, gymenv, ddpg_net_file)
