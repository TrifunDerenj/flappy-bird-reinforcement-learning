CONFIG = {
    "GAMMA": 0.99,
    "LR": 1e-3,
    "BATCH_SIZE": 64,
    "BUFFER_SIZE": 50_000,
    "MIN_REPLAY_SIZE": 1_000,
    "EPS_START": 1.0,
    "EPS_END": 0.05,
    "EPS_DECAY": 50_000,
    "TARGET_UPDATE_FREQ": 1_000,
    "CHECKPOINT_FILE": "flappy_model.pth",
    "USE_LIDAR": False,
    "RENDER": False
}
