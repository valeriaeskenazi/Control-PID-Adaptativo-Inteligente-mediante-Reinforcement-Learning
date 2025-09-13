"""
Comparison script for PPO vs DQN agents on PID controller tuning.

This script trains both agents on the same environment and provides
detailed comparison of their performance, learning curves, and behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Tuple, Any
import seaborn as sns

from environment.universal_pid_env import UniversalPIDControlEnv
from agent import create_agent
from agent.algorithms.ppo_agent import create_ppo_config
from agent.algorithms.dqn_agent import create_dqn_config


def run_training_episode(agent, env, episode_num: int) -> Dict[str, float]:
    """Run a single training episode for any agent type."""
    state = env.reset()
    total_reward = 0
    step_count = 0
    pid_parameters = []
    
    while True:
        # Get action from agent
        action = agent.select_action(state, training=True)
        pid_parameters.append(action.copy())
        
        # Take step in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Store experience
        if hasattr(agent, 'store_experience'):
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
    
    # Get final PID parameters
    final_pid = pid_parameters[-1] if pid_parameters else np.array([0, 0, 0])
    
    return {
        'total_reward': total_reward,
        'episode_length': step_count,
        'final_error': abs(info.get('error', 0)),
        'process_difficulty': info.get('process_difficulty', 'UNKNOWN'),
        'settled': abs(info.get('error', float('inf'))) <= env.dead_band,
        'final_kp': final_pid[0],
        'final_ki': final_pid[1], 
        'final_kd': final_pid[2],
        'pid_stability': np.std(pid_parameters, axis=0) if len(pid_parameters) > 1 else np.array([0, 0, 0])
    }


def run_evaluation_episode(agent, env) -> Dict[str, float]:
    """Run evaluation episode (no training)."""
    # Set evaluation mode
    if hasattr(agent, 'set_eval_mode'):
        agent.set_eval_mode()
    
    state = env.reset()
    total_reward = 0
    step_count = 0
    errors = []
    pid_parameters = []
    
    while True:
        action = agent.select_action(state, training=False)
        pid_parameters.append(action.copy())
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        errors.append(abs(info.get('error', 0)))
        state = next_state
        
        if terminated or truncated:
            break
    
    # Reset to training mode
    if hasattr(agent, 'set_train_mode'):
        agent.set_train_mode()
    
    final_pid = pid_parameters[-1] if pid_parameters else np.array([0, 0, 0])
    
    return {
        'total_reward': total_reward,
        'episode_length': step_count,
        'final_error': errors[-1] if errors else float('inf'),
        'mean_error': np.mean(errors) if errors else float('inf'),
        'settled': abs(info.get('error', float('inf'))) <= env.dead_band,
        'final_kp': final_pid[0],
        'final_ki': final_pid[1],
        'final_kd': final_pid[2],
        'error_stability': np.std(errors) if errors else 0
    }


def train_agent(agent_name: str, agent, env, num_episodes: int, update_interval: int) -> Dict[str, List]:
    """Train a single agent and return metrics."""
    print(f"\n🤖 Training {agent_name}...")
    
    metrics = {
        'rewards': [],
        'lengths': [],
        'errors': [],
        'settled_episodes': [],
        'kp_values': [],
        'ki_values': [], 
        'kd_values': [],
        'losses': [],
        'exploration_metrics': []
    }
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        episode_results = run_training_episode(agent, env, episode)
        
        # Store metrics
        metrics['rewards'].append(episode_results['total_reward'])
        metrics['lengths'].append(episode_results['episode_length'])
        metrics['errors'].append(episode_results['final_error'])
        metrics['settled_episodes'].append(1 if episode_results['settled'] else 0)
        metrics['kp_values'].append(episode_results['final_kp'])
        metrics['ki_values'].append(episode_results['final_ki'])
        metrics['kd_values'].append(episode_results['final_kd'])
        
        # Update agent periodically
        if (episode + 1) % update_interval == 0:
            update_metrics = agent.update()
            if update_metrics:
                metrics['losses'].append(update_metrics)
                
                # Store exploration metrics (different for PPO vs DQN)
                if hasattr(agent, 'get_epsilon'):
                    metrics['exploration_metrics'].append(agent.get_epsilon())
                elif 'entropy' in update_metrics:
                    metrics['exploration_metrics'].append(update_metrics['entropy'])
        
        # Print progress
        if (episode + 1) % 10 == 0:
            recent_reward = np.mean(metrics['rewards'][-10:])
            recent_settled = np.mean(metrics['settled_episodes'][-10:])
            
            exploration_info = ""
            if hasattr(agent, 'get_epsilon'):
                exploration_info = f"ε={agent.get_epsilon():.3f}"
            elif metrics['losses'] and 'entropy' in metrics['losses'][-1]:
                exploration_info = f"H={metrics['losses'][-1]['entropy']:.3f}"
            
            print(f"  Episode {episode+1:3d}: Reward={recent_reward:6.1f}, "
                  f"Settled={recent_settled:.0%}, {exploration_info}")
    
    training_time = time.time() - start_time
    print(f"  ✅ {agent_name} training completed in {training_time:.1f}s")
    
    return metrics


def evaluate_agents(agents: Dict[str, Any], env, num_eval_episodes: int = 20) -> Dict[str, Dict[str, float]]:
    """Evaluate all agents."""
    print(f"\n🧪 Evaluating agents ({num_eval_episodes} episodes each)...")
    
    eval_results = {}
    
    for name, agent in agents.items():
        print(f"  Evaluating {name}...")
        
        episode_results = []
        for _ in range(num_eval_episodes):
            result = run_evaluation_episode(agent, env)
            episode_results.append(result)
        
        # Aggregate results
        eval_results[name] = {
            'mean_reward': np.mean([r['total_reward'] for r in episode_results]),
            'std_reward': np.std([r['total_reward'] for r in episode_results]),
            'mean_error': np.mean([r['final_error'] for r in episode_results]),
            'std_error': np.std([r['final_error'] for r in episode_results]),
            'success_rate': np.mean([r['settled'] for r in episode_results]),
            'mean_episode_length': np.mean([r['episode_length'] for r in episode_results]),
            'mean_kp': np.mean([r['final_kp'] for r in episode_results]),
            'mean_ki': np.mean([r['final_ki'] for r in episode_results]),
            'mean_kd': np.mean([r['final_kd'] for r in episode_results]),
            'kp_std': np.std([r['final_kp'] for r in episode_results]),
            'ki_std': np.std([r['final_ki'] for r in episode_results]),
            'kd_std': np.std([r['final_kd'] for r in episode_results])
        }
    
    return eval_results


def plot_comparison(training_metrics: Dict[str, Dict], eval_results: Dict[str, Dict], save_path: str):
    """Create comprehensive comparison plots."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    colors = {'PPO': 'blue', 'DQN': 'red'}
    
    # Plot 1: Training rewards
    ax = axes[0, 0]
    for agent_name, metrics in training_metrics.items():
        rewards = metrics['rewards']
        episodes = range(len(rewards))
        
        ax.plot(episodes, rewards, alpha=0.3, color=colors[agent_name], linewidth=1)
        
        # Moving average
        window = min(20, len(rewards) // 10)
        if len(rewards) > window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), moving_avg, 
                   color=colors[agent_name], linewidth=2, label=f'{agent_name} (MA)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Progress: Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Success rate
    ax = axes[0, 1] 
    for agent_name, metrics in training_metrics.items():
        settled = metrics['settled_episodes']
        window = 20
        if len(settled) > window:
            success_rate = np.convolve(settled, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(settled)), success_rate,
                   color=colors[agent_name], linewidth=2, label=agent_name)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (20-ep MA)')
    ax.set_title('Success Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: PID parameter evolution - Kp
    ax = axes[0, 2]
    for agent_name, metrics in training_metrics.items():
        kp_values = metrics['kp_values']
        episodes = range(len(kp_values))
        ax.scatter(episodes[::10], kp_values[::10], alpha=0.6, s=10, 
                  color=colors[agent_name], label=f'{agent_name} Kp')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Kp Value')
    ax.set_title('Proportional Gain Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final evaluation comparison
    ax = axes[0, 3]
    agent_names = list(eval_results.keys())
    rewards = [eval_results[name]['mean_reward'] for name in agent_names]
    reward_stds = [eval_results[name]['std_reward'] for name in agent_names]
    
    bars = ax.bar(agent_names, rewards, yerr=reward_stds, 
                 color=[colors[name] for name in agent_names], alpha=0.7)
    ax.set_ylabel('Mean Evaluation Reward')
    ax.set_title('Final Performance Comparison')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, reward in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{reward:.1f}', ha='center', va='bottom')
    
    # Plot 5: Episode lengths
    ax = axes[1, 0]
    for agent_name, metrics in training_metrics.items():
        lengths = metrics['lengths']
        window = 20
        if len(lengths) > window:
            length_ma = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(lengths)), length_ma,
                   color=colors[agent_name], linewidth=2, label=agent_name)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length (20-ep MA)')
    ax.set_title('Episode Length Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Error comparison
    ax = axes[1, 1]
    agent_names = list(eval_results.keys())
    errors = [eval_results[name]['mean_error'] for name in agent_names]
    error_stds = [eval_results[name]['std_error'] for name in agent_names]
    
    bars = ax.bar(agent_names, errors, yerr=error_stds,
                 color=[colors[name] for name in agent_names], alpha=0.7)
    ax.set_ylabel('Final Error')
    ax.set_title('Control Accuracy')
    ax.grid(True, alpha=0.3)
    
    # Plot 7: PID parameters comparison
    ax = axes[1, 2]
    width = 0.25
    x = np.arange(len(agent_names))
    
    kp_means = [eval_results[name]['mean_kp'] for name in agent_names]
    ki_means = [eval_results[name]['mean_ki'] for name in agent_names]
    kd_means = [eval_results[name]['mean_kd'] for name in agent_names]
    
    ax.bar(x - width, kp_means, width, label='Kp', alpha=0.7)
    ax.bar(x, ki_means, width, label='Ki', alpha=0.7)
    ax.bar(x + width, kd_means, width, label='Kd', alpha=0.7)
    
    ax.set_xlabel('Agent')
    ax.set_ylabel('PID Parameter Value')
    ax.set_title('Final PID Parameters')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Success rate comparison
    ax = axes[1, 3]
    success_rates = [eval_results[name]['success_rate'] for name in agent_names]
    
    bars = ax.bar(agent_names, success_rates, 
                 color=[colors[name] for name in agent_names], alpha=0.7)
    ax.set_ylabel('Success Rate')
    ax.set_title('Control Success Rate')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.0%}', ha='center', va='bottom')
    
    # Plot 9: Training losses
    ax = axes[2, 0]
    for agent_name, metrics in training_metrics.items():
        if metrics['losses']:
            episodes = range(0, len(metrics['losses']) * 10, 10)  # Assuming updates every 10 episodes
            
            if agent_name == 'PPO':
                policy_losses = [l.get('policy_loss', 0) for l in metrics['losses']]
                ax.plot(episodes, policy_losses, color=colors[agent_name], 
                       linewidth=2, label=f'{agent_name} Policy Loss')
            elif agent_name == 'DQN':
                q_losses = [l.get('q_loss', 0) for l in metrics['losses']]
                ax.plot(episodes, q_losses, color=colors[agent_name],
                       linewidth=2, label=f'{agent_name} Q Loss')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 10: Exploration metrics
    ax = axes[2, 1]
    for agent_name, metrics in training_metrics.items():
        if metrics['exploration_metrics']:
            episodes = range(0, len(metrics['exploration_metrics']) * 10, 10)
            
            if agent_name == 'DQN':
                ax.plot(episodes, metrics['exploration_metrics'], 
                       color=colors[agent_name], linewidth=2, label=f'{agent_name} Epsilon')
                ax.set_ylabel('Epsilon')
            elif agent_name == 'PPO':
                ax.plot(episodes, metrics['exploration_metrics'],
                       color=colors[agent_name], linewidth=2, label=f'{agent_name} Entropy')
                ax.set_ylabel('Entropy')
    
    ax.set_xlabel('Episode')
    ax.set_title('Exploration Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 11: PID parameter stability
    ax = axes[2, 2]
    width = 0.25
    x = np.arange(len(agent_names))
    
    kp_stds = [eval_results[name]['kp_std'] for name in agent_names]
    ki_stds = [eval_results[name]['ki_std'] for name in agent_names] 
    kd_stds = [eval_results[name]['kd_std'] for name in agent_names]
    
    ax.bar(x - width, kp_stds, width, label='Kp Std', alpha=0.7)
    ax.bar(x, ki_stds, width, label='Ki Std', alpha=0.7)
    ax.bar(x + width, kd_stds, width, label='Kd Std', alpha=0.7)
    
    ax.set_xlabel('Agent')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('PID Parameter Consistency')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 12: Summary table
    ax = axes[2, 3]
    ax.axis('off')
    
    # Create summary table
    table_data = []
    for agent_name in agent_names:
        row = [
            agent_name,
            f"{eval_results[agent_name]['mean_reward']:.1f}",
            f"{eval_results[agent_name]['success_rate']:.0%}",
            f"{eval_results[agent_name]['mean_error']:.3f}",
            f"{eval_results[agent_name]['mean_episode_length']:.0f}"
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data,
                    colLabels=['Agent', 'Reward', 'Success', 'Error', 'Length'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    ax.set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main comparison script."""
    print("🚀 PPO vs DQN Comparison for PID Controller Tuning")
    print("=" * 60)
    
    # Create environment
    env = UniversalPIDControlEnv()
    print(f"Environment: {env.__class__.__name__}")
    
    # Training parameters
    num_episodes = 200
    update_interval = 10
    
    # Create agent configurations
    ppo_config = create_ppo_config(
        learning_rate=3e-4,
        hidden_dims=[128, 64],
        clip_ratio=0.2,
        ppo_epochs=4,
        mini_batch_size=64,
        gamma=0.99,
        device='cpu'
    )
    
    dqn_config = create_dqn_config(
        learning_rate=1e-3,
        hidden_dims=[128, 64],
        discretization_levels=(6, 6, 6),
        epsilon=1.0,
        epsilon_decay=0.995,
        target_update_freq=500,
        batch_size=64,
        gamma=0.99,
        double_dqn=True,
        device='cpu'
    )
    
    # Create agents
    agents = {
        'PPO': create_agent('ppo', ppo_config),
        'DQN': create_agent('dqn', dqn_config)
    }
    
    print(f"PPO Agent: Continuous actions, {sum(p.numel() for p in agents['PPO'].actor.parameters()):,} parameters")
    print(f"DQN Agent: {agents['DQN'].num_actions} discrete actions, {sum(p.numel() for p in agents['DQN'].q_network.parameters()):,} parameters")
    
    # Train agents
    training_metrics = {}
    
    for agent_name, agent in agents.items():
        metrics = train_agent(agent_name, agent, env, num_episodes, update_interval)
        training_metrics[agent_name] = metrics
    
    # Evaluate agents
    eval_results = evaluate_agents(agents, env, num_eval_episodes=30)
    
    # Print results
    print(f"\n📊 Final Evaluation Results:")
    print("-" * 60)
    for agent_name, results in eval_results.items():
        print(f"{agent_name}:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Success Rate: {results['success_rate']:.0%}")
        print(f"  Final Error: {results['mean_error']:.4f} ± {results['std_error']:.4f}")
        print(f"  Episode Length: {results['mean_episode_length']:.0f}")
        print(f"  PID Params: Kp={results['mean_kp']:.3f}, Ki={results['mean_ki']:.3f}, Kd={results['mean_kd']:.3f}")
        print()
    
    # Determine winner
    ppo_reward = eval_results['PPO']['mean_reward']
    dqn_reward = eval_results['DQN']['mean_reward']
    
    if abs(ppo_reward - dqn_reward) < 0.1:  # Close performance
        winner = "🤝 TIE"
        winner_msg = "Both agents performed similarly"
    elif ppo_reward > dqn_reward:
        winner = "🏆 PPO WINS"
        winner_msg = f"PPO outperformed DQN by {ppo_reward - dqn_reward:.2f} reward points"
    else:
        winner = "🏆 DQN WINS"
        winner_msg = f"DQN outperformed PPO by {dqn_reward - ppo_reward:.2f} reward points"
    
    print(f"🎯 FINAL RESULT: {winner}")
    print(f"   {winner_msg}")
    
    # Create comparison plots
    plot_comparison(training_metrics, eval_results, 'ppo_vs_dqn_comparison.png')
    print(f"\n📈 Detailed comparison plots saved to 'ppo_vs_dqn_comparison.png'")


if __name__ == "__main__":
    main()