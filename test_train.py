from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

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
