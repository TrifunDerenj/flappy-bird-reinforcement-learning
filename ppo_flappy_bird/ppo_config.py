CONFIG = {
    "GAMMA": 0.99,
    "LR": 3e-4,
    "CLIP_EPSILON": 0.2,
    "ENTROPY_COEF": 0.01,
    "VALUE_COEF": 0.5,
    "GAE_LAMBDA": 0.95,
    "EPOCHS_PER_UPDATE": 4,
    "BATCH_SIZE": 64,
    "BUFFER_SIZE": 2048,
    "MAX_EPISODES": 10000,
    "CHECKPOINT_FILE": "ppo_model.pth",
    "USE_LIDAR": False,
    "RENDER": False
}