import neat
import gymnasium as gym
import flappy_bird_gymnasium
import pickle
import os
import sys
import os

# Add the current directory to Python path so it can find neat_config
sys.path.append(os.path.dirname(__file__))

from neat_config import CONFIG

class NEATModel:
    def __init__(self):
        # Convert our Python dict to NEAT config file
        self._create_config_file()
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 'neat-config')
        self.genome = None
        self.network = None
    
    def _create_config_file(self):
        """Convert our Python config to NEAT's required file format"""
        config_lines = []
        
        # Convert each section
        for section_name, section_data in CONFIG.items():
            if isinstance(section_data, dict):
                config_lines.append(f"[{section_name}]")
                for key, value in section_data.items():
                    config_lines.append(f"{key} = {value}")
                config_lines.append("")
        
        # Write to file
        with open('neat-config', 'w') as f:
            f.write('\n'.join(config_lines))
    
    def load(self, checkpoint_file):
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"NEAT checkpoint not found: {checkpoint_file}")
        
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
        return action, output[0]
    
    def evaluate(self, env=None, render=False, max_steps=5000):
        """Evaluate the model for one episode"""
        if self.network is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        close_env = False
        if env is None:
            env = gym.make("FlappyBird-v0", 
                          render_mode="human" if render else None,
                          use_lidar=False)
            close_env = True
        
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action, _ = self.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        if close_env:
            env.close()
        
        return total_reward

def save_checkpoint(file, population, generation, best_fitness, best_genome):
    checkpoint_data = {
        "population": population,
        "generation": generation,
        "best_fitness": best_fitness,
        "best_genome": best_genome
    }
    with open(file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"NEAT checkpoint: gen {generation}, fitness {best_fitness:.2f}")

def load_checkpoint(file):
    with open(file, 'rb') as f:
        checkpoint_data = pickle.load(f)
    return (checkpoint_data["population"], checkpoint_data["generation"],
            checkpoint_data["best_fitness"], checkpoint_data["best_genome"])