import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class MLP():
    def __init__(self, layers):
        self.layers = layers

    def data_in(self, batch):
        #if batch['data'].shape[1] != self.input['units']:
            #return "shape error"
        self.data = batch['data']
        self.labels = batch['labels']


    def build_network(self, X, y):
        """Building the structure of the neural network"""
        self.data_in({'data': X, 'labels': y})
        self.A = [self.data]

        ### Weights and Biases initialization ###
        self.W = []
        self.B = []
        for i in range(len(self.layers) - 1):
           self.W.append(np.random.rand(self.layers[i]['units'], self.layers[i + 1]['units']))
           self.B.append(np.zeros((1, self.layers[i + 1]['units'])))
        ######
        for i in range(1, len(self.layers)):
            self.A.append(0)


        
    def forward_pass(self):
        for i in range(len(self.layers) - 1):
            Z = np.dot(self.W[i], self.A[i]) + B[i]
            activation = get_activation(self.layers[i + 1]['activation'])
            self.A[i + 1] = activation(Z)

    def backward_pass(self):
        pass


    



if __name__ == '__main__':
    net = MLP([
    {'units': 4},
    {'units': 5, 'activation': 'relu'}, 
    {'units': 8, 'activation': 'relu'},
    {'units': 1, 'activation': 'sigmoid'}
    ])

    iris = load_iris()
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    net.build_network(X_train, y_train)
