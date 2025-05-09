from setuptools import setup, find_packages

setup(
    name="nlp_gym",
    version="0.1.0",
    description="NLPGym - A toolkit for evaluating RL agents on Natural Language Processing Tasks",
    author="Rajkumar Ramamurthy",
    author_email="raj1514@gmail.com",
    packages=find_packages(),
    python_requires='>=3.7',
    url="https://github.com/rajcscw/nlp-gym/",
    install_requires=[
        "numpy", "torch", "nltk", "flair", "pandas", "matplotlib",
        "seaborn", "scikit-learn", "tqdm", "edit_distance", "rich",
        "gym", "pytorch-nlp", "datasets", "wget", "flair[word-embeddings]",
        "shimmy>=2.0"
    ],
    extras_require={
        "demo": ["tensorflow", "stable-baselines3[extra,mpi]"],
        "test": ["pytest"]
    },
)
