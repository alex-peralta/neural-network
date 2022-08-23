
import numpy as np
import matplotlib.pyplot as plt



train_images_np=np.load('./train_and_test_data/MNIST_train_images.npy')
train_labels_np=np.load('./train_and_test_data/MNIST_train_labels.npy')
val_images_np=np.load('./train_and_test_data/MNIST_val_images.npy')
val_labels_np=np.load('./train_and_test_data/MNIST_val_labels.npy')
test_images_np=np.load('./train_and_test_data/MNIST_test_images.npy')
test_labels_np=np.load('./train_and_test_data/MNIST_test_labels.npy')


N_EPOCHS=100
BATCH_SIZE = 256
TRAINING_NUM = 50000
VALIDATION_SET_SIZE = 5000
TEST_SET_SIZE = 5000
IMAGE_PIXEL_LENGTH = 28
IMAGE_PIXEL_WIDTH = 28
INPUT_LAYER_NUM = IMAGE_PIXEL_LENGTH * IMAGE_PIXEL_WIDTH
HIDDEN_LAYER_NUM = 64
OUTPUT_LAYER_NUM = 10
INITIAL_LEARNING_RATE = 1e-3
ADAPTIVE_LEARNING_PERCENTAGE = 0.97



accuracy_training_set = np.zeros(N_EPOCHS)
accuracy_validation_set = np.zeros(N_EPOCHS)

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

    '''
    The network was initialized by creating W1 and W2 weight matrices and setting all values inside those matrices to
    random gaussians with a standard deviation of 0.1. Bias vectors b1 and b2 were initialized to zero
    '''
    def __init__(self):
        mu, sigma = 0, 0.1
        # W1 is a 64x784 matrix of weights connecting the input layer and the hidden layer.
        self.W1= np.random.normal(mu, sigma, HIDDEN_LAYER_NUM * INPUT_LAYER_NUM).reshape(HIDDEN_LAYER_NUM, INPUT_LAYER_NUM)
        # b1 is a  61x1 vector of biases corresponding to the hidden layer activations.
        self.b1= np.zeros(HIDDEN_LAYER_NUM)
        # W2 is a 10x64 matrix of weights connecting the hidden layer and the output layer.
        self.W2= np.random.normal(mu, sigma, OUTPUT_LAYER_NUM * HIDDEN_LAYER_NUM).reshape(OUTPUT_LAYER_NUM,HIDDEN_LAYER_NUM)
        # b2 is a 10x1 vector of biases corresponding to the output layer activations.
        self.b2= np.zeros(OUTPUT_LAYER_NUM)
        #self.loss = 0
        self.reset_grad()

    def reset_grad(self):
        self.W2_grad = 0
        self.b2_grad = 0
        self.W1_grad = 0
        self.b1_grad = 0
        #self.loss = 0

    '''
    The forward pass of the network takes a 784x1 image vector as input, which we’ll call x. It multiples W1*x and adds 
    that to the b1 bias vector to form a 64x1 activation vector a1, where each component of the vector is one of the 64 
    hidden layer activations. The a1 vector is passed through a sigmoid function with the output being represented by a 
    64x1 vector named f1. This vector is multiplied by the second weight matrix, W2 * f1, and added to the b2 bias 
    vector the for a 10x1 activation vector a2. The a2 vector is passed through a softmax function with the output is a 
    10x1 vector named y_hat. 
    '''
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
        dLdW2 = dLdW2.reshape(OUTPUT_LAYER_NUM, HIDDEN_LAYER_NUM)
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
        dLdW1 = dLdW1.reshape(HIDDEN_LAYER_NUM, INPUT_LAYER_NUM)
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

def training_set_accuracy(epoch_num):
    correct = 0
    incorrect = 0
    for i in range(TRAINING_NUM):
        x_curr_image = train_images_np[i]
        myNet.forward(x_curr_image)
        predicted_value = np.argmax(myNet.y_hat)
        actual_value = train_labels_np[i]
        if predicted_value == actual_value:
            correct += 1
        else:
            incorrect += 1
    percentage_correct = 100.00 * correct / (correct + incorrect)
    accuracy_training_set[epoch_num] = percentage_correct

def validation_set_accuracy(epoch_num):
    correct = 0
    incorrect = 0
    for i in range(VALIDATION_SET_SIZE):
        x_curr_image = val_images_np[i]
        myNet.forward(x_curr_image)
        predicted_value = np.argmax(myNet.y_hat)
        actual_value = val_labels_np[i]
        if predicted_value == actual_value:
            correct += 1
        else:
            incorrect += 1
    percentage_correct = 100.00 * correct / (correct + incorrect)
    accuracy_validation_set[epoch_num] = percentage_correct

def train_validate_network():
    learning_rate = INITIAL_LEARNING_RATE
    # counter is used to keep of batch size
    counter = 0
    myNet.reset_grad()

    # this for loop iterates for a desired number of epochs. the larger number of epochs the more
    # accurate the network becomes, but the training time takes longer.
    for epoch_num in range(N_EPOCHS):
        #Code to train network goes here
        print(epoch_num)


        # For each epoch, random_arr shuffles numbers 1 through TRAINING_NUM in order for the network to train on
        # the images in a random order
        random_arr = np.arange(TRAINING_NUM)
        np.random.shuffle(random_arr)

        # this for loop iterates over the training set
        for r in range(TRAINING_NUM):

            # random ordered numbers corresponding to images are accessed one by one over the for loop and set to
            # curr_index. This insures all images are used but in a random order.
            curr_index = random_arr[r]

            # the current image  goes through a forward pass of the network
            x_curr_image = train_images_np[curr_index]
            myNet.forward(x_curr_image)

            # y is a vector that has the correct value of the current image. For example, if the current image was the
            # number 3, then the y vector would be [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] where the third index is set to 1.
            y =np.zeros(OUTPUT_LAYER_NUM)
            curr_image_label = train_labels_np[curr_index]
            y[curr_image_label] = 1

            # the gradients are updated using y, the correct value of the current image.
            myNet.update_grad(y)
            counter = counter + 1

            # When the counter equals the batch size, the network parameters are updated and the
            # batch is reset.
            if counter == BATCH_SIZE:
                myNet.update_params(learning_rate)
                myNet.reset_grad()
                #print(counter)
                counter = 0

        # batch size does not go into TRAINING_NUM evenly so code below is used to update params  for the last batch
        # and reset gradients and counter for the next batch for the next epoch
        myNet.update_params(learning_rate)
        myNet.reset_grad()
        #print(counter)
        counter = 0

        #Call functions to compute accuracy of training and validation sets
        training_set_accuracy(epoch_num)
        validation_set_accuracy(epoch_num)

        # update learning rate after each epoch
        learning_rate = learning_rate * ADAPTIVE_LEARNING_PERCENTAGE

def accuracy_and_confusion_matrix():
    conf_matrx = np.zeros(OUTPUT_LAYER_NUM**2).reshape(OUTPUT_LAYER_NUM, OUTPUT_LAYER_NUM)
    correct = 0
    incorrect = 0
    for x in range(TEST_SET_SIZE):
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

    conf_matrx2 = np.zeros(shape=(OUTPUT_LAYER_NUM,OUTPUT_LAYER_NUM))
    counter = 0
    for row in conf_matrx:
        SUM = sum(row)
        row = 100 * row / SUM
        conf_matrx2[counter] = row
        counter += 1

    print(conf_matrx2)

def training_and_testing_graph():
    x_axis_arr = np.arange(N_EPOCHS)
    plt.ylim([75,100])
    plt.plot(x_axis_arr, accuracy_training_set, label="Training")
    plt.plot(x_axis_arr, accuracy_validation_set, label="Validation")
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

def weight_visualization_W1():
    W1 = np.load('W1.npy')
    visual = np.zeros(shape=(IMAGE_PIXEL_LENGTH,IMAGE_PIXEL_WIDTH))

    print(W1.shape)
    counter = 0
    for row in W1:
        if counter >= 0 and counter < 4:
            visual = row.reshape(IMAGE_PIXEL_LENGTH,IMAGE_PIXEL_WIDTH)
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

    correct = 0
    incorrect = 0
    for x in range(TEST_SET_SIZE):
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

train_validate_network()

accuracy_and_confusion_matrix()

training_and_testing_graph()

save_weights_biases()

weight_visualization_W1()

check_accuracy_from_upload()
