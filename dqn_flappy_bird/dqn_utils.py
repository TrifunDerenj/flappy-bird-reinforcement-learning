import torch
import collections
import os

def save_checkpoint(file, policy_net, target_net, optimizer, buffer, eps, total_steps, episode):
    """Save DQN checkpoint"""
    torch.save({
        "policy_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "replay_buffer": list(buffer.buffer),
        "eps": eps,
        "total_steps": total_steps,
        "episode": episode
    }, file)
    print(f"Checkpoint saved at episode {episode}, step {total_steps}")

def load_checkpoint(file, policy_net, target_net, optimizer, buffer):
    """Load DQN checkpoint"""
    torch.serialization.add_safe_globals([collections.deque])
    
    checkpoint = torch.load(file, weights_only=False)
    policy_net.load_state_dict(checkpoint["policy_net"])
    target_net.load_state_dict(checkpoint["target_net"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    if isinstance(checkpoint["replay_buffer"], collections.deque):
        buffer.buffer = checkpoint["replay_buffer"]
    else:
        buffer.buffer = collections.deque(checkpoint["replay_buffer"], maxlen=buffer.buffer.maxlen)
    
    return checkpoint["eps"], checkpoint["total_steps"], checkpoint["episode"]