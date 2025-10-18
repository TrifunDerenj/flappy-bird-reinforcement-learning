import os
import random
import time
import torch
import gymnasium as gym
import flappy_bird_gymnasium
import sys
sys.path.append("..")

from dqn_config import CONFIG
from replay_buffer import ReplayBuffer
from dqn_model import DQN
from dqn_utils import save_checkpoint, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fill_replay_buffer(env, buffer, min_size):
    obs, _ = env.reset()
    for _ in range(min_size):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()[0]

def train_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma):
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    states_v = torch.FloatTensor(states).to(device)
    actions_v = torch.LongTensor(actions).to(device)
    rewards_v = torch.FloatTensor(rewards).to(device)
    next_states_v = torch.FloatTensor(next_states).to(device)
    dones_v = torch.BoolTensor(dones).to(device)

    q_values = policy_net(states_v).gather(1, actions_v.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = target_net(next_states_v).max(1)[0]
        next_q_values[dones_v] = 0.0
        target_q_values = rewards_v + gamma * next_q_values

    loss = torch.nn.functional.mse_loss(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=CONFIG["USE_LIDAR"])
    obs, _ = env.reset()
    obs_size = len(obs)
    n_actions = env.action_space.n

    policy_net = DQN(obs_size, n_actions).to(device)
    target_net = DQN(obs_size, n_actions).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=CONFIG["LR"])
    replay_buffer = ReplayBuffer(CONFIG["BUFFER_SIZE"])

    eps = CONFIG["EPS_START"]
    total_steps = 0
    episode = 0

    if os.path.exists(CONFIG["CHECKPOINT_FILE"]):
        eps, total_steps, episode = load_checkpoint(CONFIG["CHECKPOINT_FILE"], policy_net, target_net, optimizer, replay_buffer)
        print(f"Resumed from episode {episode}, step {total_steps}")
    else:
        print("Filling replay buffer with random actions...")
        fill_replay_buffer(env, replay_buffer, CONFIG["MIN_REPLAY_SIZE"])

    print("Training DQN for 10 minutes...")
    start_time = time.time()
    
    try:
        while time.time() - start_time < 600:  # 10 minutes
            obs, _ = env.reset()
            episode_reward = 0
            done = False

            while not done and (time.time() - start_time < 600):
                if random.random() < eps:
                    action = env.action_space.sample()
                else:
                    state_v = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    with torch.no_grad():
                        action = int(torch.argmax(policy_net(state_v), dim=1).item())

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay_buffer.push(obs, action, reward, next_obs, done)
                obs = next_obs
                episode_reward += reward

                if len(replay_buffer) > CONFIG["MIN_REPLAY_SIZE"]:
                    train_step(policy_net, target_net, optimizer, replay_buffer,
                               CONFIG["BATCH_SIZE"], CONFIG["GAMMA"])

                if total_steps % CONFIG["TARGET_UPDATE_FREQ"] == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                eps = max(CONFIG["EPS_END"], CONFIG["EPS_START"] - total_steps / CONFIG["EPS_DECAY"])
                total_steps += 1

            print(f"Episode {episode+1}: reward = {episode_reward:.2f}, eps = {eps:.3f}")
            episode += 1

    except KeyboardInterrupt:
        print("\nInterrupted!")

    print("\nSaving DQN checkpoint...")
    save_checkpoint(CONFIG["CHECKPOINT_FILE"], policy_net, target_net, optimizer,
                    replay_buffer, eps, total_steps, episode)
    env.close()
    print("DQN training completed!")

if __name__ == "__main__":
    main()
