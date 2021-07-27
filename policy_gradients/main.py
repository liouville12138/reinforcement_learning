import gym
import os
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt


def update(module, env, file_saved):
    step = 0
    step_old = 0
    avg_step = 0
    for episode in range(10000):
        # initial observation
        observation = env.reset()
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = module.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)

            module.store_transition(observation, action, reward)

            if done:
                ep_rs_sum = sum(module.reward)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                print("episode:", episode, "  reward:", int(running_reward))

                vt = module.learn()

                print("Episode {} finished after {} timesteps".format(episode, step - step_old))
                break
            # swap observation
            observation = observation_
            # break while loop when end of this episode
            step += 1
        avg_step = ((step - step_old) * 0.05 + avg_step * 0.95)
        print(avg_step)
        step_old = step

    # end of game
    module.save_net(file_saved)
    print('game over')


if __name__ == "__main__":
    # maze game
    gymenv = gym.make('CartPole-v0')
    gymenv = gymenv.unwrapped  # 放开步数限制
    RL = PolicyGradient(gymenv.action_space.n,
                        gymenv.observation_space.shape[0],
                        learning_rate=0.01,
                        batch_size=32,
                        gamma=0.9,
                        )

    policy_gradient_net = "policy_gradient_net.pickle"
    actions = list(range(gymenv.action_space.n))

    if os.path.exists(policy_gradient_net):
        RL.load_net(policy_gradient_net)
    update(RL, gymenv, policy_gradient_net)
