import time
import numpy as np


class DNN:
    def __init__(self, sizes=[784, 128, 64, 10], epochs=10, lr=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr

        input_layer = sizes[0]
        hidden_layer_1 = sizes[1]
        hidden_layer_2 = sizes[2]
        output_layer = sizes[3]

        self.params = {
            # 128x784
            'W1': np.random.randn(hidden_layer_1, input_layer) * np.sqrt(1./hidden_layer_1),

            # 64x128
            'W2': np.random.randn(hidden_layer_2, hidden_layer_1) * np.sqrt(1./hidden_layer_2),

            # 10x64
            'W3': np.random.randn(output_layer, hidden_layer_2) * np.sqrt(1./output_layer)
        }

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1+np.exp(-x))

    def softmax(self, x, derivative=False):
        exps = np.exp(x-x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1-exps / np.sum(exps, axis=0))
        return exps/np.sum(exps, axis=0)

    def forward_pass(self, x_train):
        params = self.params

        params['A0'] = x_train  # 784x1

        # input_layer to hidden_layer_1
        params['Z1'] = np.dot(params['W1'], params['A0'])  # 128x1
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden_layer_1 to hidden_layer_2
        params['Z2'] = np.dot(params['W2'], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden_layer_2 to output_layer
        params['Z3'] = np.dot(params['W3'], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['Z3']

    def backward_pass(self, y_train, output):
        params = self.params

        change_w = {}

        # calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # calculate W2 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def update_weights(self, change_w):
        for key, val in change_w.items():
            self.params[key] -= self.lr * val #W_t+1 = W_t - lr*Delta_W_t

    def compute_accuracy(self, test_data):
        predictions = []
        for x in test_data:
            values = x.split(",")
            inputs = (np.asarray(values[1:], dtype=float) / 255.0 *0.99) + 0.01
            targets = np.zeros(10) + 0.01
            targets[int(values[0])] = 0.99
            output = self.forward_pass(inputs)
            pred = np.argmax(output)
            predictions.append(pred==np.argmax(targets))
        return np.mean(predictions)

    def train(self, train_list, test_list):
        start_time = time.time()
        for i in range(self.epochs):
            for x in train_list:
                values = x.split(",")
                inputs = (np.asarray(values[1:], dtype=float) / 255.0 * 0.99) + 0.01
                targets = np.zeros(10) + 0.01
                targets[int(values[0])] = 0.99
                output = self.forward_pass(inputs)
                change_w = self.backward_pass(targets, output)
                self.update_weights(change_w)
            accuracy = self.compute_accuracy(test_list)
            print('Epoch {0}, Time Spent: {1:.02f}s, Accuracy: {2:.2f}%'.format(i+1, time.time()-start_time, accuracy*100))

train_file = open("/home/suito/Documents/Projet/CNN_MNIST/train.csv")
train_list = train_file.readlines()
train_file.close

test_file = open("/home/suito/Documents/Projet/CNN_MNIST/test.csv")
test_list = test_file.readlines()
test_file.close

dnn = DNN(sizes=[784, 256, 128, 10], epochs=20, lr=0.001)
dnn.train(train_list, test_list)
