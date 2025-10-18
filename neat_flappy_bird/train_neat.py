import os
import time
import pickle
import gymnasium as gym
import flappy_bird_gymnasium
import neat
from neat_config import CONFIG
from neat_model import NEATModel, save_checkpoint, load_checkpoint

def eval_genome(genome, config):
    """Evaluate a single genome"""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make("FlappyBird-v0", use_lidar=CONFIG["USE_LIDAR"])
    obs, _ = env.reset()
    fitness = 0
    done = False
    steps = 0
    
    while not done and steps < 5000:
        output = net.activate(obs)
        action = 1 if output[0] > 0.5 else 0
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        fitness += reward
        steps += 1
    
    env.close()
    # Bonus for surviving longer
    fitness += steps * 0.1
    return fitness

def eval_genomes(genomes, config):
    """Evaluate all genomes in the population"""
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def main():
    neat_model = NEATModel()
    
    # Create population and set up reporters
    population = neat.Population(neat_model.config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    generation = 0
    best_fitness = -float('inf')
    best_genome = None
    start_time = time.time()

    # Load checkpoint if exists
    if os.path.exists(CONFIG["CHECKPOINT_FILE"]):
        population, generation, best_fitness, best_genome = load_checkpoint(CONFIG["CHECKPOINT_FILE"])
        print(f"Resumed from generation {generation}, best fitness {best_fitness:.2f}")

    print("Training NEAT for 10 minutes...")

    try:
        # Use NEAT's built-in run method with timeout
        def eval_function(genomes, config):
            return eval_genomes(genomes, config)
        
        # Run for generations until time runs out
        while time.time() - start_time < 600:  # 10 minutes
            # Run one generation
            population.run(eval_function, 1)
            generation += 1
            
            # Get the best genome from this generation
            best_genome = stats.best_genome()
            if best_genome and best_genome.fitness > best_fitness:
                best_fitness = best_genome.fitness
                print(f"Generation {generation}: NEW BEST! fitness = {best_fitness:.2f}")
            
            # Save checkpoint
            save_checkpoint(CONFIG["CHECKPOINT_FILE"], population, generation, best_fitness, best_genome)
            
            # Check time
            if time.time() - start_time >= 600:
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted!")
    
    # Save final model
    if best_genome:
        with open('neat_winner.pkl', 'wb') as f:
            pickle.dump(best_genome, f)
        print(f"NEAT training completed! Best fitness: {best_fitness:.2f}")
    else:
        print("NEAT training completed with no valid genomes.")

if __name__ == "__main__":
    main()