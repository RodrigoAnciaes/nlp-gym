# NLPGym - Modified Educational Version [![CircleCI](https://circleci.com/gh/rajcscw/nlp-gym/tree/main.svg?style=svg)](https://circleci.com/gh/rajcscw/nlp-gym/tree/main)

> **Note**: This is a **modified fork** of the original [nlp-gym](https://github.com/rajcscw/nlp-gym) repository, adapted for educational purposes with updated dependencies and compatibility fixes.

NLPGym is a toolkit to bridge the gap between applications of RL and NLP. This aims at facilitating research and benchmarking of DRL application on natural language processing tasks. 

The toolkit provides interactive environments for standard NLP tasks such as sequence tagging, question answering, and sequence classification.

## Educational Content

This repository includes educational materials in the `classes/` directory, organized in multiple parts:

- **Part 1** (`part1.ipynb`): Introduction to Reinforcement Learning fundamentals and Q-Learning
- **Part 2** (`part2.ipynb`): Deep Reinforcement Learning with Neural Networks and DQN
- **Part 3** (`part3.ipynb`): Embeddings and appliyng them in with NLP_gym)
- **Practical Application** (`nlpgym_practical_application.ipynb`): Complete hands-on guide with working examples

## Modifications from Original

This fork includes several important modifications to ensure compatibility with modern libraries:

### Dependency Updates
- **Stable Baselines**: Updated to work with both Stable-Baselines (v2.10.2) and Stable-Baselines3 (v2.6.0)
- **Gymnasium Compatibility**: Added wrappers to bridge old Gym API with new Gymnasium API
- **Library Compatibility**: Fixed deprecated functions and updated import statements for newer versions of key dependencies

### Key Changes
- Fixed tokenization issues in sequence tagging environments
- Updated featurizers to work with current Flair versions
- Added compatibility layers for modern PyTorch and transformers
- Resolved numpy version conflicts
- Updated dataset loading to work with recent Hugging Face datasets

### Demo Scripts
- Enhanced training scripts with better error handling
- Added comprehensive evaluation and plotting capabilities
- Included CPU-optimized configurations for resource-limited environments

## Tasks Supported by NLP_GYM

Sequence Tagging             |  Question Answering |  Multi-label Classification
:-------------------------:|:-------------------------:|:-------------------------:
<img src="assets/sequence_tagging.png" width="100%"/> |  <img src="assets/question_answering.png" width="100%"/> |  <img src="assets/multilabel.png" width="100%"/> 

- **Sequence Tagging:** Sequence tagging task can be cast as an MDP in which the given sentence is parsed in left-to-right order. At each step, one token is presented to the agent. The actions available to the agent are to TAG with one of the possible labels. The episode terminates when the end of the sentence is reached. By default, reward function is based on entity level F1 scores. It can be either *sparse* given at the end of the episode or *dense* in which at each step, a change in scores between steps is given as reward.

- **Multiple-Choice Question Answering (QA):** The task of QA is to answer a given question by selecting one of the multiple choices. Questions are often accompanied by supporting facts which contain further context. Selecting the correct option out of all choices can be considered as a sequential decision-making task. At each step, the observation consists of question, facts and a choice. The available actions are (i) ANSWER and (ii) CONTINUE. On ANSWER, the last shown choice is considered as the selection choice and the episode terminates. On CONTINUE, next observation is shown with a different choice. The reward is given only at the end of the episode, either 0 or 1, based on the selected choice's correctness.

- **Multi-Label Classification:** Multi-label classification is a generalization of several NLP tasks such as multi-class sentence classification and label ranking. The task of multi-label classification is to assign a label sequence to the given sentence. In information retrieval, this task corresponds to label ranking when preferential relation exists over labels. Likewise, the task reduces to a simple multi-class classification when any label sequence's maximum length is at most one. In any case, generating this label sequence can be cast as a sequential decision-making task. Similar to sequence tagging, available actions are to INSERT one of the possible labels. Moreover, agents can terminate the episode using the TERMINATE action

The environments provide standard RL interfaces and therefore can be used together with most RL frameworks such as [baselines](https://github.com/openai/baselines), [stable-baselines](https://github.com/hill-a/stable-baselines), and [RLLib](https://github.com/ray-project/ray). 

Furthermore, the toolkit is designed in a modular fashion providing flexibility for users to extend tasks with their custom data sets, observations, and reward functions.

## Installation

### From source (recommended for this modified version):

```bash
git clone https://github.com/YourUsername/nlp-gym.git
cd nlp-gym
pip install .
```

For the full demo experience with all algorithms:
```bash
git clone https://github.com/YourUsername/nlp-gym.git
cd nlp-gym
pip install .["demo"]
```

### PyTorch with CUDA support:
```bash
# For CUDA 11.8 (adjust version as needed)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Visit [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your specific system configuration.

## Quick Start

### Basic Usage of NLP_GYM Example

```python
from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv

# Data pool
pool = QASC.prepare("train")

# Question answering environment
env = QAEnv()
for sample, weight in pool:
    env.add_sample(sample)

# Play an episode
done = False
state = env.reset()
total_reward = 0
while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    total_reward += reward
    env.render()
    print(f"Action: {env.action_space.ix_to_action(action)}")
print(f"Episodic reward {total_reward}")
```

### Training with DQN (Stable-Baselines3)

```python
from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer
from stable_baselines3 import DQN
import gymnasium as gym

# Compatibility wrapper for Gymnasium
class CompatEnv(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = self._convert_space(env.action_space)
        self.observation_space = self._convert_space(env.observation_space)
    
    def _convert_space(self, space):
        # Convert old gym spaces to gymnasium spaces
        if hasattr(space, 'n'):
            return gym.spaces.Discrete(space.n)
        elif hasattr(space, 'low'):
            return gym.spaces.Box(space.low, space.high, dtype=space.dtype)
        return space
    
    def reset(self, **kwargs):
        obs = self.env.reset()
        return obs, {}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info

# Prepare environment
data_pool = QASC.prepare(split="train")
featurizer = InformedFeaturizer()
env = QAEnv(observation_featurizer=featurizer)

for sample, weight in data_pool:
    env.add_sample(sample, weight)

# Wrap for compatibility
compat_env = CompatEnv(env)

# Train DQN agent
model = DQN("MlpPolicy", compat_env, gamma=0.99, batch_size=32, 
            learning_rate=1e-4, verbose=1)
model.learn(total_timesteps=int(1e4))
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
2. **GPU Issues**: The code works on CPU by default. For GPU support, install PyTorch with CUDA
3. **Memory Issues**: Use the provided CPU-optimized configurations in the demo scripts
4. **Environment Compatibility**: Use the provided wrapper classes for Gymnasium compatibility

For more examples and detailed documentation, explore the training scripts in [demo_scripts](https://github.com/rajcscw/nlp-gym/tree/main/demo_scripts) and the educational notebooks in the `classes/` directory.
