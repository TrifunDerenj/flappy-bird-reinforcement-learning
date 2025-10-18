import os
import time
import subprocess
import sys
from pathlib import Path

# Training time in seconds (10 minutes)
TRAINING_TIME = 10 * 60

# Model configurations - using your existing directory structure
MODEL_CONFIGS = {
    'dqn': {
        'dir': 'dqn_flappy_bird',
        'script': 'train_dqn.py',
        'config': 'dqn_config.py'
    },
    'ppo': {
        'dir': 'ppo_flappy_bird', 
        'script': 'train_ppo.py',
        'config': 'ppo_config.py'
    },
    'neat': {
        'dir': 'neat_flappy_bird',
        'script': 'train_neat.py', 
        'config': 'neat_config.py'
    }
}

def check_files_exist():
    """Check if all required files exist"""
    print("Checking for required files...")
    all_files_exist = True
    
    for model_type, model_info in MODEL_CONFIGS.items():
        model_dir = Path(model_info['dir'])
        train_script = model_dir / model_info['script']
        config_file = model_dir / model_info['config']
        
        print(f"\n{model_type.upper()}:")
        if train_script.exists():
            print(f"Training script: {train_script}")
        else:
            print(f"Missing training script: {train_script}")
            all_files_exist = False
            
        if config_file.exists():
            print(f"Config file: {config_file}")
        else:
            print(f"Missing config file: {config_file}")
            all_files_exist = False
            
        # Check for other required files in each directory
        other_files = list(model_dir.glob("*.py"))
        other_files = [f for f in other_files if f.name not in [model_info['script'], model_info['config']]]
        if other_files:
            print(f"Other files: {[f.name for f in other_files]}")
    
    return all_files_exist

def run_training(model_type):
    """Run training for a specific model for 10 minutes"""
    model_info = MODEL_CONFIGS[model_type]
    model_dir = Path(model_info['dir'])
    train_script = model_dir / model_info['script']
    
    if not train_script.exists():
        print(f"Training script not found: {train_script}")
        return False
    
    print(f"\n{'='*60}")
    print(f"STARTING {model_type.upper()} TRAINING FOR 10 MINUTES")
    print(f"{'='*60}")
    start_time = time.time()
    
    try:
        # Run the training script
        print(f"Running: python {train_script.name}")
        result = subprocess.run(
            [sys.executable, str(train_script.name)],
            cwd=str(model_dir),
            timeout=TRAINING_TIME + 30,  # Add 30 seconds grace period
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"{model_type.upper()} training completed successfully")
            # Show relevant output
            lines = result.stdout.strip().split('\n')
            print("Last output lines:")
            for line in lines[-10:]:  # Show last 10 lines
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"✗ {model_type.upper()} training failed with return code {result.returncode}")
            if result.stderr:
                print("Error output:")
                for line in result.stderr.strip().split('\n')[-10:]:
                    if line.strip():
                        print(f"  {line}")
            
    except subprocess.TimeoutExpired:
        print(f"{model_type.upper()} training stopped after 10 minutes")
    except Exception as e:
        print(f"{model_type.upper()} training failed: {e}")
        return False
    
    elapsed = time.time() - start_time
    print(f"\n{model_type.upper()} training took {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
    return True

def main():
    """Main function to run all trainings"""
    print("FLAPPY BIRD RL TRAINING - 10 MINUTES PER MODEL")
    print("=" * 60)
    
    # Check if files exist
    if not check_files_exist():
        print("\nSome required files are missing. Please check the errors above.")
        print("Make sure you have all the training scripts in their respective directories.")
        return
    
    print("\nAll required files found!")
    print(f"Each model will train for {TRAINING_TIME//60} minutes")
    
    # Ask for confirmation
    input("\nPress Enter to start training or Ctrl+C to cancel...")
    
    # Run trainings
    print("\n" + "=" * 60)
    print("STARTING TRAINING SEQUENCE")
    print("=" * 60)
    
    results = {}
    for model_type in ['dqn', 'ppo', 'neat']:
        print(f"\n>>> Preparing {model_type.upper()}...")
        success = run_training(model_type)
        
        results[model_type] = success
        
        if model_type != 'neat':  # Don't wait after the last one
            print(f"\nWaiting 5 seconds before next model...")
            time.sleep(5)
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    successful_models = [model for model, success in results.items() if success]
    failed_models = [model for model, success in results.items() if not success]
    
    if successful_models:
        print(f"Successful: {', '.join(successful_models).upper()}")
    if failed_models:
        print(f"Failed: {', '.join(failed_models).upper()}")
    
    # Show checkpoints
    print("\nCheckpoint files created:")
    for model_type, model_info in MODEL_CONFIGS.items():
        model_dir = Path(model_info['dir'])
        checkpoint_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pkl"))
        if checkpoint_files:
            print(f"  {model_type.upper()}:")
            for cf in checkpoint_files:
                size_kb = cf.stat().st_size / 1024
                print(f"    - {cf.name} ({size_kb:.1f} KB)")
        else:
            print(f"  {model_type.upper()}: No checkpoints found")

if __name__ == "__main__":
    main()