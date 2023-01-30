# differential-privacy-ml
Implementation of Differentially Private Machine Learning

## Files in this repository
```
Root
│   .gitignore
│   custom_logging.py
│   README.md
│   requirements.txt
│
├───baseline_pytorch
│   │   mnist.py
│   │   mnist_datasets.py
│   │
│   ├───data
│   │   └───MNIST
│   │       └───raw
│   └───saved
│           model.pth
│           optimizer.pth
│
├───baseline_tf
│       mnist.py
├───dp_sgd_opacus
│   │   mnist.py
│   │   mnist_datasets.py
│   │
│   └───saved
│           model.pth
│           optimizer.pth
│  
└───existing-code
        .gitkeep
        mnist-dp (1).ipynb
```

## How to run

To run the pytorch baseline:

    1. Clone this repository 
    2. Create a virtual env or activate an existing env
    3. Install dependencies using: `pip install -r requirements.txt`
    4. Go to the baseline directory using: `cd baseline_pytorch`
    5. Run the script using: `python mnist.py`


To run the tensorflow baseline:

    1. Clone this repository 
    2. Create a virtual env or activate an existing env
    3. Install dependencies using: `pip install -r requirements.txt`
    4. Go to the baseline directory using: `cd baseline_tf`
    5. Run the script using: `python mnist.py`


To run the pytorch-opacus dp baseline:

    1. Clone this repository 
    2. Create a virtual env or activate an existing env
    3. Install dependencies using: `pip install -r requirements.txt`
    4. Go to the baseline directory using: `cd dp_sgd_opacus`
    5. Run the script using: `python mnist.py`

## Results

### Pytorch Baselines

#### Non-DP results:
Train Accuracy: 99.99%
Test Accuracy: 99.99%

#### DP-SGD results:

MAX_GRAD_NORM = 4  
DELTA = 1e-5  
MOMENTUM = 0.9  
BACTH_SIZE = 600  
EPOCHS = 100      
LEARNING_RATE = 0.05  


| Epsilon | Sigma    |Train Loss | Train Accuracy | Test Loss | Test Accuracy |
|---------|----------|-----------|----------------|-----------|---------------|
|    1    |          |           |                |           |               |
|    2    | 2.13867  |  0.7625   |     87.85      |  0.7388   |    88.59      |
|    5    | 1.12548  |  0.4183   |     90.85      |  0.3925   |    91.50      |
|    10   | 0.80230  |  0.3590   |     91.27      |  0.3448   |    92.12      |
|    20   | 0.62690  |  0.3327   |     91.70      |  0.3241   |    92.54      |
|         |          |           |                |           |               |


| Epsilon | Train Loss | Test Loss | Test Accuracy |
| 0.1     | 0.0000     | 0.0000    | 0.0000        |
| 1.0     | 0.0000     | 0.0000    | 0.0000        |
| 2.0     | 1.2646     | 1.1823    | 87.86         |
| 4.99    | 0.6454     | 0.5952    | 90.99         |
| 10.0    | 0.0000     | 0.0000    | 0.0000        |
| 15.0    | 0.0000     | 0.0000    | 0.0000        |
| 50.0    | 0.0000     | 0.0000    | 0.0000        |



