from setuptools import setup, find_packages

setup(
    name="attention-guided-nca",
    version="0.1.0",
    description="Neural Cellular Automaton with Convolutional Attention",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.46.0",
        "pillow>=7.1.2",
    ],
    python_requires=">=3.7",
    extras_require={
        "dev": [
            "pytest>=6.0.0",
        ],
    },
) 