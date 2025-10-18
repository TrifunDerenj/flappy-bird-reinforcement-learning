import neat
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import flappy_bird_gymnasium
import os
import collections
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN network
class DQN(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(obs_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# PPO network
class PPONetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(PPONetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)

# NEAT model wrapper
class NEATModel:
    def __init__(self):
        import neat
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 'neat_flappy_bird/neat-config')
        self.genome = None
        self.network = None
    
    def load(self, checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        self.genome = checkpoint_data["best_genome"]
        self.network = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        return checkpoint_data["generation"], checkpoint_data["best_fitness"]
    
    def predict(self, state):
        if self.network is None:
            raise ValueError("Model not loaded")
        output = self.network.activate(state)
        action = 1 if output[0] > 0.5 else 0
        return action

def load_dqn_model(checkpoint_file, obs_size, n_actions, device):
    """Load DQN model safely"""
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"DQN checkpoint not found: {checkpoint_file}")
    
    model = DQN(obs_size, n_actions).to(device)
    torch.serialization.add_safe_globals([collections.deque])
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    
    if "policy_net" in checkpoint:
        model.load_state_dict(checkpoint["policy_net"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def load_ppo_model(checkpoint_file, obs_size, n_actions, device):
    """Load PPO model"""
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint_file}")
    
    model = PPONetwork(obs_size, n_actions).to(device)
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def load_neat_model(checkpoint_file):
    """Load NEAT model"""
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"NEAT checkpoint not found: {checkpoint_file}")
    
    model = NEATModel()
    model.load(checkpoint_file)
    return model

def show_menu():
    """Display model selection menu"""
    print("\n" + "="*50)
    print("FLAPPY BIRD AI DEMO")
    print("="*50)
    print("Choose which AI model to watch:")
    print("1. DQN (Deep Q-Network)")
    print("2. PPO (Proximal Policy Optimization)") 
    print("3. NEAT (NeuroEvolution of Augmenting Topologies)")
    print("4. Exit")
    print("="*50)
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            else:
                print("Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            return '4'

def play_model(model_type, model, env, obs_size, n_actions):
    """Play the game with the selected model"""
    print(f"\nStarting {model_type} Flappy Bird AI...")
    print("Press Ctrl+C to return to menu")
    
    episode_count = 0
    total_reward = 0
    
    try:
        while True:
            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                if model_type == "NEAT":
                    # NEAT model
                    action = model.predict(obs)
                else:
                    # DQN or PPO model
                    state_v = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    with torch.no_grad():
                        if model_type == "DQN":
                            q_values = model(state_v)
                            action = int(torch.argmax(q_values, dim=1).item())
                        else:  # PPO
                            logits, _ = model(state_v)
                            action = int(torch.argmax(logits, dim=1).item())

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            episode_count += 1
            total_reward += episode_reward
            print(f"Episode {episode_count}: {model_type} score = {episode_reward}, total = {total_reward}")

    except KeyboardInterrupt:
        print(f"\n{model_type} session ended.")
        print(f"Final stats: {episode_count} episodes, average score: {total_reward/episode_count:.2f}")

def main():
    # Initialize environment to get observation size
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    obs, _ = env.reset()
    obs_size = len(obs)
    n_actions = env.action_space.n
    env.close()
    
    # Model file paths
    model_files = {
        '1': ("DQN", "dqn_flappy_bird/flappy_model.pth"),
        '2': ("PPO", "ppo_flappy_bird/ppo_model.pth"), 
        '3': ("NEAT", "neat_flappy_bird/neat_model.pkl")
    }
    
    while True:
        choice = show_menu()
        
        if choice == '4':
            print("Goodbye!")
            break
            
        model_type, model_file = model_files[choice]
        
        try:
            # Load the selected model
            if choice == '1':  # DQN
                model = load_dqn_model(model_file, obs_size, n_actions, device)
            elif choice == '2':  # PPO  
                model = load_ppo_model(model_file, obs_size, n_actions, device)
            else:  # NEAT
                model = load_neat_model(model_file)
            
            # Create environment and play
            env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
            play_model(model_type, model, env, obs_size, n_actions)
            env.close()
            
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("Make sure the model files exist in the correct directories:")
            print("  - dqn_flappy_bird/flappy_model.pth")
            print("  - ppo_flappy_bird/ppo_model.pth") 
            print("  - neat_flappy_bird/neat_model.pkl")
            input("\nPress Enter to continue...")
            
        except Exception as e:
            print(f"\nError loading {model_type} model: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()