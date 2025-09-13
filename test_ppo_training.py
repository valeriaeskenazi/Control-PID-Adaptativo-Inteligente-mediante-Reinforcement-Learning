"""
Simple training loop to test PPO agent with the universal PID environment.

This script demonstrates end-to-end training using all our modular components:
- UniversalPIDControlEnv from environment/
- PPOAgent from agent/algorithms/
- All the supporting infrastructure (buffers, policies, networks)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from environment.universal_pid_env import UniversalPIDControlEnv
from agent import create_agent
from agent.algorithms.ppo_agent import create_ppo_config


def run_training_episode(agent, env, episode_num: int) -> Dict[str, float]:
    """Run a single training episode."""
    state = env.reset()
    total_reward = 0
    step_count = 0
    
    while True:
        # Get action from agent
        action = agent.select_action(state, training=True)
        
        # Take step in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Store experience in agent's buffer
        agent.store_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=terminated or truncated
        )
        
        total_reward += reward
        step_count += 1
        state = next_state
        
        if terminated or truncated:
            break
    
    return {
        'total_reward': total_reward,
        'episode_length': step_count,
        'final_error': abs(info.get('error', 0)),
        'process_difficulty': info.get('process_difficulty', 'UNKNOWN'),
        'settled': abs(info.get('error', float('inf'))) <= env.dead_band
    }


def run_evaluation_episode(agent, env) -> Dict[str, float]:
    """Run a single evaluation episode (no training)."""
    agent.set_eval_mode()
    
    state = env.reset()
    total_reward = 0
    step_count = 0
    errors = []
    
    while True:
        # Get action from agent (no exploration)
        action = agent.select_action(state, training=False)
        
        # Take step in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        errors.append(abs(info.get('error', 0)))
        state = next_state
        
        if terminated or truncated:
            break
    
    agent.set_train_mode()
    
    return {
        'total_reward': total_reward,
        'episode_length': step_count,
        'final_error': errors[-1] if errors else float('inf'),
        'mean_error': np.mean(errors) if errors else float('inf'),
        'settled': abs(info.get('error', float('inf'))) <= env.dead_band
    }


def main():
    """Main training loop."""
    print("🤖 Starting PPO Training for PID Controller Tuning")
    print("=" * 60)
    
    # Create environment
    env = UniversalPIDControlEnv()
    print(f"Environment created: {env.__class__.__name__}")
    
    # Create PPO agent configuration
    config = create_ppo_config(
        learning_rate=3e-4,
        hidden_dims=[128, 64],
        clip_ratio=0.2,
        ppo_epochs=4,  # Reduced for faster training
        mini_batch_size=32,
        buffer_capacity=2048,
        gamma=0.99,
        gae_lambda=0.95,
        device='cpu'
    )
    
    # Create agent
    agent = create_agent('ppo', config)
    print(f"Agent created: {agent.__class__.__name__}")
    print(f"PID ranges: Kp{config.kp_range}, Ki{config.ki_range}, Kd{config.kd_range}")
    
    # Training parameters
    num_episodes = 100
    eval_interval = 10
    update_interval = 10  # Update every N episodes
    
    # Metrics storage
    training_rewards: List[float] = []
    training_lengths: List[int] = []
    evaluation_rewards: List[float] = []
    training_losses: List[Dict[str, float]] = []
    
    print(f"\nStarting training for {num_episodes} episodes...")
    print("Episode | Reward  | Length | Error   | Difficulty | Update")
    print("-" * 60)
    
    # Training loop
    for episode in range(num_episodes):
        # Run training episode
        episode_results = run_training_episode(agent, env, episode)
        
        training_rewards.append(episode_results['total_reward'])
        training_lengths.append(episode_results['episode_length'])
        
        # Print progress
        print(f"{episode:7d} | {episode_results['total_reward']:6.1f} | "
              f"{episode_results['episode_length']:6d} | "
              f"{episode_results['final_error']:6.3f} | "
              f"{episode_results['process_difficulty']:10s} | ", end="")
        
        # Update agent periodically
        if (episode + 1) % update_interval == 0:
            update_metrics = agent.update()
            training_losses.append(update_metrics)
            
            if update_metrics:
                print(f"Policy: {update_metrics.get('policy_loss', 0):.3f}")
            else:
                print("No update")
        else:
            print("---")
        
        # Run evaluation periodically
        if (episode + 1) % eval_interval == 0:
            eval_results = run_evaluation_episode(agent, env)
            evaluation_rewards.append(eval_results['total_reward'])
            
            print(f"    🎯 Evaluation: Reward={eval_results['total_reward']:.1f}, "
                  f"Error={eval_results['final_error']:.3f}, "
                  f"Settled={'✅' if eval_results['settled'] else '❌'}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    
    # Print final statistics
    recent_rewards = training_rewards[-10:]
    print(f"\nFinal Performance:")
    print(f"  Average reward (last 10): {np.mean(recent_rewards):.2f}")
    print(f"  Best reward: {np.max(training_rewards):.2f}")
    print(f"  Average episode length: {np.mean(training_lengths):.1f}")
    
    if evaluation_rewards:
        print(f"  Average evaluation reward: {np.mean(evaluation_rewards):.2f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training rewards
    plt.subplot(2, 3, 1)
    plt.plot(training_rewards, alpha=0.7, label='Episode Reward')
    if len(training_rewards) > 10:
        # Moving average
        window = min(10, len(training_rewards) // 10)
        moving_avg = np.convolve(training_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(training_rewards)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Episode lengths
    plt.subplot(2, 3, 2)
    plt.plot(training_lengths, alpha=0.7, color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Evaluation rewards
    if evaluation_rewards:
        plt.subplot(2, 3, 3)
        eval_episodes = list(range(eval_interval-1, num_episodes, eval_interval))
        plt.plot(eval_episodes, evaluation_rewards, 'go-', linewidth=2, markersize=8)
        plt.xlabel('Episode')
        plt.ylabel('Evaluation Reward')
        plt.title('Evaluation Performance')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Loss curves
    if training_losses:
        plt.subplot(2, 3, 4)
        policy_losses = [loss.get('policy_loss', 0) for loss in training_losses]
        value_losses = [loss.get('value_loss', 0) for loss in training_losses]
        
        update_episodes = list(range(update_interval-1, len(policy_losses) * update_interval, update_interval))
        
        plt.plot(update_episodes, policy_losses, 'b-', label='Policy Loss', linewidth=2)
        plt.plot(update_episodes, value_losses, 'r-', label='Value Loss', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 5: Entropy
    if training_losses:
        plt.subplot(2, 3, 5)
        entropies = [loss.get('entropy', 0) for loss in training_losses]
        plt.plot(update_episodes, entropies, 'purple', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Policy Entropy')
        plt.title('Exploration (Entropy)')
        plt.grid(True, alpha=0.3)
    
    # Plot 6: Agent info
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, "PPO Agent Configuration:", fontsize=12, fontweight='bold')
    plt.text(0.1, 0.7, f"Learning Rate: {config.learning_rate}")
    plt.text(0.1, 0.6, f"Clip Ratio: {config.clip_ratio}")
    plt.text(0.1, 0.5, f"PPO Epochs: {config.ppo_epochs}")
    plt.text(0.1, 0.4, f"Mini Batch Size: {config.mini_batch_size}")
    plt.text(0.1, 0.3, f"Network: {config.hidden_dims}")
    plt.text(0.1, 0.2, f"Gamma: {config.gamma}")
    plt.text(0.1, 0.1, f"GAE Lambda: {config.gae_lambda}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Configuration')
    
    plt.tight_layout()
    plt.savefig('ppo_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nResults saved to 'ppo_training_results.png'")
    
    # Test final trained agent
    print(f"\n🧪 Testing final trained agent:")
    final_test = run_evaluation_episode(agent, env)
    print(f"  Final test reward: {final_test['total_reward']:.2f}")
    print(f"  Final test error: {final_test['final_error']:.3f}")
    print(f"  System settled: {'✅' if final_test['settled'] else '❌'}")
    
    # Show sample PID parameters
    sample_state = np.array([50.0, 50.0, 0.0, 0.0, 0.0, 0.0])  # Perfect control state
    sample_action = agent.select_action(sample_state, training=False)
    print(f"  Sample PID params for perfect state: Kp={sample_action[0]:.3f}, Ki={sample_action[1]:.3f}, Kd={sample_action[2]:.3f}")


if __name__ == "__main__":
    main()