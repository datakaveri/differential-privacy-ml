from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input 
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


model = Sequential()
model.add(Dense(1000, activation='relu', input_shape=(60,)))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(X_train_pca, y_train, epochs=100, batch_size=32, validation_data=(X_test_pca, y_test))