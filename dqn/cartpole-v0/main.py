import gym
import os
from RL_brain import DeepQNetwork


def update(module, env, file_saved):
    step = 0
    step_old = 0
    avg_step = 0
    for episode in range(10000):
        # initial observation
        observation = env.reset()
        ep_r = 0

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = module.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            module.store_transition(observation, action, r, observation_)

            ep_r += r
            if module.memory_counter > module.memory_size:
                module.learn()
                if done:
                    print('Ep: ', episode, '| Ep_r: ', round(ep_r, 2))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            step += 1
            if done:
                print("Episode {} finished after {} timesteps".format(episode, step - step_old))
                break
        avg_step = ((step - step_old) * 0.05 + avg_step * 0.95)
        step_old = step

    # end of game
    module.save_net(file_saved)
    print('game over')


if __name__ == "__main__":
    # maze game
    gymenv = gym.make('CartPole-v0')
    gymenv = gymenv.unwrapped  # 放开步数限制
    RL = DeepQNetwork(gymenv.action_space.n,
                      gymenv.observation_space.shape[0],
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      batch_size=32,
                      )

    q_table_file = "dqn_net.pickle"
    actions = list(range(gymenv.action_space.n))

    if os.path.exists(q_table_file):
        RL.load_net(q_table_file)
    update(RL, gymenv, q_table_file)
