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


To run the pytorch-opcaus dp baseline:

    1. Clone this repository 
    2. Create a virtual env or activate an existing env
    3. Install dependencies using: `pip install -r requirements.txt`
    4. Go to the baseline directory using: `cd dp_sgd_opacus`
    5. Run the script using: `python mnist.py`