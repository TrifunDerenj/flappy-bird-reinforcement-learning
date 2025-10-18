import torch
import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import pickle
import collections
import matplotlib.pyplot as plt
from dqn_flappy_bird.dqn_model import DQN
from ppo_flappy_bird.ppo_model import PPONetwork
from neat_flappy_bird.neat_model import NEATModel

def evaluate_unified_models(num_episodes=10):
    env = gym.make("FlappyBird-v0", use_lidar=False)
    obs, _ = env.reset()
    obs_size = len(obs)
    n_actions = env.action_space.n
    device = torch.device("cpu")
    
    results = {}
    
    # Evaluate DQN
    try:
        print("Evaluating DQN...")
        model = DQN(obs_size, n_actions).to(device)
        torch.serialization.add_safe_globals([collections.deque])
        checkpoint = torch.load("dqn_flappy_bird/flappy_model.pth", map_location=device, weights_only=False)  # FIXED PATH
        if "policy_net" in checkpoint:
            model.load_state_dict(checkpoint["policy_net"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        
        rewards = []
        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                state = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(state)
                    action = torch.argmax(q_values, dim=1).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            rewards.append(episode_reward)
        results['DQN'] = {'mean': np.mean(rewards), 'std': np.std(rewards)}
        print(f"DQN: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    except Exception as e:
        print(f"DQN failed: {e}")
    
    # Evaluate PPO
    try:
        print("Evaluating PPO...")
        model = PPONetwork(obs_size, n_actions).to(device)
        checkpoint = torch.load("ppo_flappy_bird/ppo_model.pth", map_location=device, weights_only=False)  # FIXED PATH
        model.load_state_dict(checkpoint["model"])
        model.eval()
        
        rewards = []
        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                state = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = model(state)
                    action = torch.argmax(logits, dim=1).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            rewards.append(episode_reward)
        results['PPO'] = {'mean': np.mean(rewards), 'std': np.std(rewards)}
        print(f"PPO: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    except Exception as e:
        print(f"PPO failed: {e}")
    
    # Evaluate NEAT
    try:
        print("Evaluating NEAT...")
        neat_model = NEATModel()
        neat_model.load("neat_flappy_bird/neat_model.pkl")  # FIXED PATH
        
        rewards = []
        for _ in range(num_episodes):
            reward = neat_model.evaluate(env=env, render=False)
            rewards.append(reward)
        results['NEAT'] = {'mean': np.mean(rewards), 'std': np.std(rewards)}
        print(f"NEAT: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    except Exception as e:
        print(f"NEAT failed: {e}")
    
    env.close()
    
    # Print results
    print("\n=== FINAL RESULTS ===")
    for model, stats in results.items():
        print(f"{model}: {stats['mean']:.2f} ± {stats['std']:.2f}")
    
    # Find best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['mean'])
        print(f"\nBest performing model: {best_model[0]} with mean reward {best_model[1]['mean']:.2f}")
    
    if results:
        models = list(results.keys())
        means = [results[model]['mean'] for model in models]
        stds = [results[model]['std'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, means, yerr=stds, capsize=5, alpha=0.7, color=['blue', 'green', 'red'])
        plt.ylabel('Mean Reward')
        plt.title('Comparison of RL Algorithms on Flappy Bird')
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{mean:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.show()
        print("\nGraph saved as 'model_comparison.png'")
    
    return results

if __name__ == "__main__":
    evaluate_unified_models()