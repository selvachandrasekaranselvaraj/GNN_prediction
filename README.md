# GNN_prediction

Graph Neural Network (GNN) based prediction of structural and electronic properties of materials.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Data Processing](#data-processing)
6. [Model Architecture](#model-architecture)
7. [Training](#training)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview

GNN_prediction is a Python package designed to predict various structural and electronic properties of materials using Graph Neural Networks (GNNs). The project leverages data from the Materials Project database to create graph representations of crystal structures and then uses these graphs to train GNN models for property prediction.

The properties predicted include:
- Volume (V)
- Number of atoms (n)
- Density (ρ)
- Bulk modulus (B1, B2, B3)
- Shear modulus (G1, G2, G3)
- Coordination number (CN)
- Space group number (SGN)
- Energy per atom (UE, E)
- Formation energy per atom (FE)
- Energy above hull (EAH)
- Stability (is_S)
- Band gap (Eg)
- Conduction band minimum (CBM)
- Valence band maximum (VBM)
- Fermi energy (Ef)

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/GNN_prediction.git
   cd GNN_prediction
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\\Scripts\\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

Note: Make sure you have Python 3.7+ installed.

## Usage

The project workflow consists of several steps:

1. Downloading structures from the Materials Project
2. Converting structures to graphs
3. Analyzing the dataset statistics
4. Training GNN models for property prediction

To run the prediction process, use the `run_prediction.py` script:

```
python scripts/run_prediction.py
```

This script will process the data, generate graphs, and perform cross-validation using the components from the `gnn_prediction` package.

## Project Structure

The project is organized as follows:

```
GNN_prediction/
├── gnn_prediction/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── graph_generation.py
│   ├── model.py
│   ├── train_test.py
│   ├── utils.py
│   └── main.py
├── scripts/
│   └── run_prediction.py
├── tests/
│   └── __init__.py
├── README.md
├── setup.py
└── LICENSE.md
```

## Data Processing

### Downloading Structures

To download structures from the Materials Project:

1. Set your Materials Project API key in the `MAPI_KEY` variable in `gnn_prediction/data_processing.py`.
2. Run the script:
   ```
   python gnn_prediction/data_processing.py
   ```

This will download structures for different crystal systems and save them as pickle files.

### Converting Structures to Graphs

To convert the downloaded structures to graph representations:

1. Run the `graph_generation.py` script:
   ```
   python gnn_prediction/graph_generation.py
   ```

This script will process the structures and create graph representations, saving them in the `graphs/` directory.

### Analyzing Dataset Statistics

To analyze the statistics of the dataset:

1. Run the `data_processing.py` script:
   ```
   python gnn_prediction/data_processing.py
   ```

This will generate various plots and statistics about the dataset, saving them in the `results_plots/` directory.

## Model Architecture

The GNN model is defined in the `GNN` class in `gnn_prediction/model.py`. It consists of:

- Multiple Graph Convolutional Network (GCN) layers
- Global mean pooling
- A final linear layer for prediction

The model architecture can be customized by adjusting the `hidden_dim` and `num_layers` parameters.

## Training

To train the GNN models:

1. Run the `train_test.py` script:
   ```
   python gnn_prediction/train_test.py
   ```

This script will:
- Load the graph data
- Perform cross-validation for each property
- Train GNN models
- Save the best models in the `trained_models/` directory
- Generate training plots and save them in the `results_plots/` directory

You can adjust training parameters such as `epochs`, `batch_size`, `learning_rate`, and `patience` in the `cross_validate` function.

## Results

The training results for each property are saved as CSV files and plots in the `results_plots/` directory. These include:

- Training and validation loss curves
- Training and validation R² score curves
- True vs. predicted value plots

You can analyze these results to assess the model's performance for each property.

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature`)
6. Create a new Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.