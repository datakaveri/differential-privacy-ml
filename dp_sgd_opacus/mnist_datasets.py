from tensorflow import keras
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

pca = PCA(n_components=60)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

minmax = MinMaxScaler()
X_train_pca = minmax.fit_transform(X_train_pca)
X_test_pca = minmax.transform(X_test_pca)

class MNISTDatasetTrain(Dataset):
    def __init__(self):
        self.X_train_pca = X_train_pca
        self.y_train = y_train
        
    def __len__(self):
        return len(self.X_train_pca)

    def __getitem__(self, idx):
        return torch.Tensor(self.X_train_pca[idx]), torch.tensor([self.y_train[idx]])
    
class MNISTDatasetTest(Dataset):
    def __init__(self):
        self.X_test_pca = X_test_pca
        self.y_test = y_test
        
    def __len__(self):
        return len(self.X_test_pca)

    def __getitem__(self, idx):
        return torch.Tensor(self.X_test_pca[idx]), torch.tensor([self.y_test[idx]])
    


