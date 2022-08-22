
import numpy as np
import time
import matplotlib.pyplot as plt



train_images_np=np.load('./train_and_test_data/MNIST_train_images.npy')
train_labels_np=np.load('./train_and_test_data/MNIST_train_labels.npy')
val_images_np=np.load('./train_and_test_data/MNIST_val_images.npy')
val_labels_np=np.load('./train_and_test_data/MNIST_val_labels.npy')
test_images_np=np.load('./train_and_test_data/MNIST_test_images.npy')
test_labels_np=np.load('./train_and_test_data/MNIST_test_labels.npy')

learning_rate=1e-3
n_epochs=100
batch_size = 256
training_num = 50000
validation_set_size = 5000


#print(train_images_np.shape)
#print(train_labels_np.shape)
##Template MLP code
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def CrossEntropy(y_hat,y):
    return -np.dot(y,np.log(y_hat))

class MLP():

    def __init__(self):
        mu, sigma = 0, 0.1
        self.W1= np.random.normal(mu, sigma, 50176).reshape(64,784)
        self.b1= np.zeros(64)
        self.W2= np.random.normal(mu, sigma, 640).reshape(10,64)
        self.b2= np.zeros(10)
        #self.loss = 0
        self.reset_grad()

    def reset_grad(self):
        self.W2_grad = 0
        self.b2_grad = 0
        self.W1_grad = 0
        self.b1_grad = 0
        #self.loss = 0

    def forward(self, x):

        self.x=x
        self.W1x= np.dot(self.W1, self.x)
        self.a1= self.W1x + self.b1
        self.f1= sigmoid(self.a1)
        self.W2x= np.dot(self.W2, self.f1)
        self.a2= self.W2x + self.b2
        self.y_hat= softmax(self.a2)

        return self.y_hat

    def update_grad(self,y):
        #self.loss = self.loss + CrossEntropy(self.y_hat, y)
        dLdA2 = self.y_hat - y

        #dLdW2
        #dA2dW2
        dLdW2 = np.array([])
        for val in dLdA2:
            curr = val * self.f1
            dLdW2 = np.concatenate((dLdW2, curr), axis=0)
        dLdW2 = dLdW2.reshape(10, 64)
        #dLdW2 = np.transpose(dLdW2)

        # dLdW1
        # dA1dW1
        dA2dF1= self.W2
        dF1dA1= self.f1 * (1 - self.f1)
        dLdF1 = dLdA2 @ self.W2
        dLdA1 = dLdF1 * dF1dA1
        dLdW1 = np.array([])

        for val in dLdA1:
            curr = val * self.x
            dLdW1 = np.concatenate((dLdW1, curr), axis=0)
        dLdW1 = dLdW1.reshape(64, 784)
        #dLdW1 = np.transpose(dLdW1)

        # dA2db2
        dLdb2 = dLdA2

        # dA1db1
        dLdb1 = dLdA1

        self.W2_grad = self.W2_grad + dLdW2
        self.b2_grad = self.b2_grad + dLdb2
        self.W1_grad = self.W1_grad + dLdW1
        self.b1_grad = self.b1_grad + dLdb1


    def update_params(self,learning_rate):
        self.W2 = self.W2 - learning_rate * self.W2_grad
        self.b2 = self.b2 - learning_rate * self.b2_grad.reshape(-1)
        self.W1 = self.W1 - learning_rate * self.W1_grad
        self.b1 = self.b1 - learning_rate * self.b1_grad.reshape(-1)
        #self.loss = self.loss / 256
        #print(self.loss)


def train_network():
    num = 0
    myNet.reset_grad()

    for iter in range(n_epochs):
        #Code to train network goes here
        print(iter)
        random_arr = np.arange(training_num)
        np.random.shuffle(random_arr)
        for r in range(training_num):
            x = random_arr[r]

            myNet.forward(train_images_np[x])
            y =np.zeros(10)
            y[train_labels_np[x]] = 1
            myNet.update_grad(y)
            num = num + 1
            if num == batch_size:
                myNet.update_params(learning_rate)
                myNet.reset_grad()
                #print(num)
                num = 0
        myNet.update_params(learning_rate)
        myNet.reset_grad()
        #print(num)
        num = 0


        #Code to compute validation loss/accuracy goes here
        correct = 0
        incorrect = 0
        for x in range(training_num):
            myNet.forward(train_images_np[x])
            predicted_value = np.argmax(myNet.y_hat)
            actual_value = train_labels_np[x]
            if predicted_value == actual_value:
                correct += 1
            else:
                incorrect += 1
        percentage = 100.00 * correct / (correct + incorrect)
        training_line_graph_arr[iter] = percentage

        correct = 0
        incorrect = 0
        for x in range(validation_set_size):
            myNet.forward(val_images_np[x])
            predicted_value = np.argmax(myNet.y_hat)
            actual_value = val_labels_np[x]
            if predicted_value == actual_value:
                correct += 1
            else:
                incorrect += 1
        percentage = 100.00 * correct / (correct + incorrect)
        val_line_graph_arr[iter] = percentage

        learning_rate = learning_rate *0.97

def accuracy_and_confusion_matrix():
    conf_matrx = np.zeros(100).reshape(10, 10)
    test_set_size = 5000
    correct = 0
    incorrect = 0
    for x in range(test_set_size):
        myNet.forward(test_images_np[x])
        predicted_value = np.argmax(myNet.y_hat)
        actual_value = test_labels_np[x]
        conf_matrx[actual_value][predicted_value] += 1
        if predicted_value == actual_value:
            correct += 1
        else:
            incorrect += 1
    accuracy = 100.00 * correct / (correct + incorrect)

    print("accuracy")
    print(accuracy)

    conf_matrx2 = np.zeros(shape=(10,10))
    counter = 0
    for row in conf_matrx:
        SUM = sum(row)
        row = 100 * row / SUM
        conf_matrx2[counter] = row
        counter += 1

    print(conf_matrx2)

def training_and_testing_graph():
    x_axis_arr = np.arange(100)
    plt.ylim([75,100])
    plt.plot(x_axis_arr, training_line_graph_arr, label="Training")
    plt.plot(x_axis_arr, val_line_graph_arr, label="Validation")
    plt.title('Training with 50,000 Images')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right', borderaxespad=1.)
    plt.show()

def save_weights_biases():
    np.save('W1.npy', myNet.W1)
    np.save('W2.npy', myNet.W2)
    np.save('b1.npy', myNet.b1)
    np.save('b2.npy', myNet.b2)

def W1_weight_visualization():
    W1 = np.load('W1.npy')
    visual = np.zeros(shape=(28,28))

    print(W1.shape)
    counter = 0
    for row in W1:
        if counter >= 0 and counter < 4:
            visual = row.reshape(28,28)
            plt.imshow(visual)
            plt.axis('off')
            plt.show()
        counter += 1

def check_accuracy_from_upload():
    W1 = np.load('W1.npy')
    W2 = np.load('W2.npy')
    b1 = np.load('b1.npy')
    b2 = np.load('b2.npy')

    myNet = MLP()
    myNet.W1 = W1
    myNet.W2 = W2
    myNet.b1 = b1
    myNet.b2 = b2

    test_set_size = 5000
    correct = 0
    incorrect = 0
    for x in range(test_set_size):
        myNet.forward(test_images_np[x])
        predicted_value = np.argmax(myNet.y_hat)
        actual_value = test_labels_np[x]
        if predicted_value == actual_value:
            correct += 1
        else:
            incorrect += 1
    accuracy = 100.00 * correct / (correct + incorrect)

    print('Accuracy of the MLP network on the 5000 test images: %.2f %%' % (accuracy))





## Init the MLP
myNet=MLP()



#n_epochs=1
#batch_size = 1
#training_num = 1
## Training code
training_line_graph_arr = np.zeros(100)
val_line_graph_arr = np.zeros(100)


train_network()

# 2.6 and 2.7




accuracy_and_confusion_matrix()
#2.5

training_and_testing_graph()

#2.8


save_weights_biases()



W1_weight_visualization()

check_accuracy_from_upload()
