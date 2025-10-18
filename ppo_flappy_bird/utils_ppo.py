import torch
import os

def save_checkpoint(file, model, optimizer, episode, best_reward):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": episode,
        "best_reward": best_reward
    }, file)
    print(f"Checkpoint saved at episode {episode}, best reward {best_reward}")

def load_checkpoint(file, model, optimizer):
    checkpoint = torch.load(file, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["episode"], checkpoint["best_reward"]