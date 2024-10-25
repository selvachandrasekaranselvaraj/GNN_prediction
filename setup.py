from setuptools import setup, find_packages

setup(
    name="GNN_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torch_geometric",
        "pymatgen",
        "numpy",
        "pandas",
        "tqdm",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "scipy"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Graph Neural Network (GNN) based prediction of structural and electronic properties of materials",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/GNN_prediction",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)