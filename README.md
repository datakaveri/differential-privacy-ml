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
│
└───existing-code
        .gitkeep
        mnist-dp (1).ipynb
```

## How to run

To run the pytorch baseline:

    1. git clone this repository 
    2. pip install -r requirements.txt
    3. cd baseline_pytorch
    4. python mnist.py


