import gym
from dpn import Agent
from utils import plot_learning
import numpy as np
from gym import wrappers


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    lr = 0.0005
    n_games = 2000

    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        alpha=lr,
        input_dims=[8],
        n_actions=4,
        mem_size=1000000,
        n_games=n_games,
        batch_size=64
    )

    alpha = f"alpha{lr}"

    filename = f"0-lunar-lander-256x256-{alpha}-bs64-adam-faster_decay.png"
    scores = []
    eps_history = []

    score = 0
    env = wrappers.Monitor(env, "tmp/lunar-lander-4", video_callable=lambda episode_id: True, force=True)

    for i in range(n_games):
        done = False
        score_str = f"episode: {i:0>4}, score: {score:>6.4f}"
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i - 10):(i + 1)])
            print(score_str + f", average score {avg_score:6>.4f}, epsilon: {agent.epsilon:6>.4f}")
            # agent.save_models()
        else:
            print(score_str)

        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, int(done))
            observation = observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

    x = [i + 1 for i in range(n_games)]
    plot_learning(x, scores, eps_history, filename)
