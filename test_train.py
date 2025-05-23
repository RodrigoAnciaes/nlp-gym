from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

def train():
    # 1) prepare your data pools
    data_pool = QASC.prepare(split="train")
    val_pool  = QASC.prepare(split="val")
    # 2) featurizer & env setup
    featurizer = InformedFeaturizer()
    env = QAEnv(observation_featurizer=featurizer)
    for sample, weight in data_pool:
        env.add_sample(sample, weight)
    # 4) instantiate & train SB3‑DQN
    model = DQN(
        policy="MlpPolicy",
        env=env,
        gamma=0.99,
        batch_size=32,
        learning_rate=1e-4,
        exploration_fraction=0.1,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
    )
    model.learn(total_timesteps=int(1e4))
    # save the model
    models_dir = "models"
    # create the models dir if not exist
    os.makedirs(models_dir, exist_ok=True)
    model.save(f"{models_dir}/dqn_qa_model")
    return model, val_pool

def plot_learning_curve(model, val_pool):
    """
    Plot the learning curve using validation data.
    
    Args:
        model: Trained DQN model
        val_pool: Validation data pool
    """
    # Create evaluation environment
    featurizer = InformedFeaturizer()
    eval_env = QAEnv(observation_featurizer=featurizer)
    for sample, weight in val_pool:
        eval_env.add_sample(sample, weight)
    
    # Extract learning progress data from the model
    if hasattr(model, 'ep_info_buffer') and len(model.ep_info_buffer) > 0:
        rewards = [ep_info['r'] for ep_info in model.ep_info_buffer]
        episode_lengths = [ep_info['l'] for ep_info in model.ep_info_buffer]
        episodes = range(len(rewards))
        
        # Plot rewards
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(episodes, rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward per Episode')
        
        # Plot episode lengths
        plt.subplot(1, 2, 2)
        plt.plot(episodes, episode_lengths)
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.title('Episode Length')
        
        plt.tight_layout()
        plt.savefig('learning_curve.png')
        plt.show()
    else:
        print("No learning progress data available.")

def test_and_plot(model, val_pool, num_episodes=100):
    """
    Test the model on the validation set and plot performance metrics.
    
    Args:
        model: Trained DQN model
        val_pool: Validation data pool
        num_episodes: Number of episodes to evaluate
    """
    # Create evaluation environment
    featurizer = InformedFeaturizer()
    eval_env = QAEnv(observation_featurizer=featurizer)
    for sample, weight in val_pool:
        eval_env.add_sample(sample, weight)
    
    # Track performance metrics
    rewards = []
    correct_answers = 0
    question_types = defaultdict(list)
    
    # Evaluate model on validation set
    for i in tqdm(range(num_episodes), desc="Evaluating"):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            # Convert numpy array to int if needed
            if isinstance(action, np.ndarray):
                action = action.item()  # Convert single-element array to scalar
            
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward
            
            if done:
                rewards.append(episode_reward)
                if episode_reward > 0:  # Assuming positive reward means correct answer
                    correct_answers += 1
                
                # Group performance by question type if available in info
                if 'question_type' in info:
                    question_types[info['question_type']].append(episode_reward)
    
    # Calculate accuracy
    accuracy = correct_answers / num_episodes
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot reward distribution
    plt.subplot(2, 2, 1)
    plt.hist(rewards, bins=10)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title(f'Reward Distribution (Accuracy: {accuracy:.2f})')
    
    # Plot rewards by episode
    plt.subplot(2, 2, 2)
    plt.plot(range(len(rewards)), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    
    # Plot performance by question type if available
    if question_types:
        plt.subplot(2, 2, 3)
        q_types = list(question_types.keys())
        q_accuracies = [sum(r > 0 for r in rewards) / len(rewards) 
                       for q_type, rewards in question_types.items()]
        
        plt.bar(q_types, q_accuracies)
        plt.xlabel('Question Type')
        plt.ylabel('Accuracy')
        plt.title('Performance by Question Type')
        plt.xticks(rotation=45)
    
    # Show confusion matrix or common error types if available
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, f"Total Episodes: {num_episodes}\nCorrect Answers: {correct_answers}\nAccuracy: {accuracy:.4f}", 
             horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.axis('off')
    plt.title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.show()
    
    print(f"Model Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    model, val_pool = train()
    plot_learning_curve(model, val_pool)
    test_and_plot(model, val_pool)