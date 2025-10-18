import os
import time
import torch
import torch.nn.functional as F 
import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
from collections import deque

from ppo_config import CONFIG
from ppo_model import PPONetwork
from utils_ppo import save_checkpoint, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
    
    def push(self, state, action, reward, done, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
    
    def get_data(self):
        return (np.array(self.states), np.array(self.actions), np.array(self.rewards),
                np.array(self.dones), np.array(self.values), np.array(self.log_probs))

def compute_advantages(rewards, values, dones, next_value, gamma, gae_lambda):
    advantages = []
    gae = 0
    next_value = next_value
    next_not_done = 1 - dones[-1]
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value * next_not_done - values[t]
        gae = delta + gamma * gae_lambda * next_not_done * gae
        advantages.insert(0, gae)
        next_value = values[t]
        next_not_done = 1 - dones[t]
    return advantages

def train_step(model, optimizer, states, actions, old_log_probs, returns, advantages, config):
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    old_log_probs = torch.FloatTensor(old_log_probs).to(device)
    returns = torch.FloatTensor(returns).to(device)
    advantages = torch.FloatTensor(advantages).to(device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    for _ in range(config["EPOCHS_PER_UPDATE"]):
        logits, values = model(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        # Actor loss
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - config["CLIP_EPSILON"], 1 + config["CLIP_EPSILON"]) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy
        entropy_loss = -dist.entropy().mean()
        
        # Total loss
        total_loss = (actor_loss + 
                     config["VALUE_COEF"] * critic_loss + 
                     config["ENTROPY_COEF"] * entropy_loss)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    return actor_loss.item(), critic_loss.item(), entropy_loss.item()

def main():
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=CONFIG["USE_LIDAR"])
    obs, _ = env.reset()
    obs_size = len(obs)
    n_actions = env.action_space.n

    model = PPONetwork(obs_size, n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"])
    buffer = PPOBuffer()

    episode = 0
    best_reward = -float('inf')
    start_time = time.time()  # For 10-minute timing

    if os.path.exists(CONFIG["CHECKPOINT_FILE"]):
        episode, best_reward = load_checkpoint(CONFIG["CHECKPOINT_FILE"], model, optimizer)
        print(f"Resumed from episode {episode}, best reward {best_reward:.2f}")

    print("Training PPO for 10 minutes...")

    try:
        while time.time() - start_time < 600:  # 10 minutes
            obs, _ = env.reset()
            episode_reward = 0
            done = False

            while not done and (time.time() - start_time < 600):
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, value = model(state_t)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                
                buffer.push(obs, action.item(), reward, done, value.item(), log_prob.item())
                obs = next_obs
                episode_reward += reward

            if len(buffer.states) > 0:
                states, actions, rewards, dones, values, old_log_probs = buffer.get_data()
                
                with torch.no_grad():
                    next_state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    _, next_value = model(next_state_t)
                    next_value = next_value.item()
                
                advantages = compute_advantages(rewards, values, dones, next_value, 
                                              CONFIG["GAMMA"], CONFIG["GAE_LAMBDA"])
                returns = advantages + values
                
                actor_loss, critic_loss, entropy_loss = train_step(
                    model, optimizer, states, actions, old_log_probs, returns, advantages, CONFIG
                )
                buffer.clear()
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                save_checkpoint(CONFIG["CHECKPOINT_FILE"], model, optimizer, episode, best_reward)
                print(f"Episode {episode+1}: NEW BEST! reward = {episode_reward:.2f}")
            else:
                print(f"Episode {episode+1}: reward = {episode_reward:.2f}, best = {best_reward:.2f}")
            
            episode += 1

    except KeyboardInterrupt:
        print("\nInterrupted!")

    print("\nSaving final PPO checkpoint...")
    save_checkpoint(CONFIG["CHECKPOINT_FILE"], model, optimizer, episode, best_reward)
    env.close()
    print("PPO training completed!")

if __name__ == "__main__":
    main()