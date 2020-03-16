from keras.metrics import categorical_accuracy

# Modify the input shape by adding a channels-dimension in the end
def modify_input_shape(input):
    if len(input.shape) == 3:
        return input.reshape(input.shape + (1,))

# Calculate the accuracy of a classifier given examples and targets
def calc_accuracy_classifier(classifier, x_data, y_data):
    cat_acc = categorical_accuracy(classifier.predict(modify_input_shape(x_data)),y_data)
    acc = (sum(cat_acc)/len(cat_acc))*100
    print("Accuracy: ", acc.numpy(), "%")