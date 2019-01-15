import numpy as np
import pandas as pd
import tensorflow
from tensorflow.python.keras.layers import Input, Dense
from keras.layers import Dense, Dropout
from util import func_confusion_matrix
from keras.models import Sequential, Model
from keras import regularizers, metrics
from keras.utils import to_categorical

def feed_forward_model(train_X,train_y,val_x,val_y,testX,testY):
    """feed_forward_model - specification list
    Create a feed forward model given a specification list
    Each element of the list represents a layer and is formed by a tuple.

    [(Dense, [20], {'activation':'relu', 'input_dim': M}),
     (Dense, [20], {'activation':'relu', 'input_dim':20}),
     (Dense, [N], {'activation':'softmax', 'input_dim':20})
    ]
	
    """
    model = Sequential()
    model_list = [  [(Dense, [100],
                    {'activation': 'relu', 'input_dim': train_X.shape[1]}),
                   (Dense, [200], {'activation': 'relu', 'input_dim': 100}),
                    (Dense, [200], {'activation': 'relu', 'input_dim': 100}),
                   (Dense, [2], {'activation': 'softmax', 'input_dim':100})],

                [(Dense, [500],
                  {'activation': 'relu', 'input_dim': train_X.shape[1]}),
                 (Dense, [1000], {'activation': 'relu', 'input_dim': 100}),
                 (Dense, [2], {'activation': 'softmax', 'input_dim': 100})],

                [(Dense, [100],
                  {'activation': 'relu', 'input_dim': train_X.shape[1]}),
                 (Dense, [100], {'activation': 'relu', 'input_dim': 100}),
                 (Dense, [100], {'activation': 'relu', 'input_dim': 100}),
                 (Dense, [100], {'activation': 'relu', 'input_dim': 100}),
                 (Dense, [100], {'activation': 'relu', 'input_dim': 100}),
                 (Dense, [100], {'activation': 'relu', 'input_dim': 100}),
                 (Dense, [100], {'activation': 'relu', 'input_dim': 100}),
                     (Dense, [100], {'activation': 'relu', 'input_dim': 100}),
                 (Dense, [100], {'activation': 'relu', 'input_dim': 100}),
                 (Dense, [2], {'activation': 'softmax', 'input_dim': 100})],
                ]
				
    for item in model_list[2]:
        layertype = item[0]
        if (len(item) < 3):
            layer = layertype(*item[1])
        else:
            layer = layertype(*item[1], **item[2])
        model.add(layer)

    model.compile(optimizer="Adam",
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    model.fit(train_X, to_categorical(train_y), verbose=0)
    model_eval_result = model.evaluate(val_x,to_categorical(val_y), verbose=False)
    print("Loss value", model_eval_result[0])
    print("Accuracy", model_eval_result[1])
	#prediction = tf.argmax(logits,1)
    y_pred_nn = model.predict(testX)
    y_pred_nn_val = np.argmax(y_pred_nn.round(), axis=1)
    y_pred=model.predict(testX)
    correct_labels = 0
    for i in range(testX.shape[0]):
        if(testY[i] == y_pred_nn_val[i]):
            correct_labels += 1

    #accuracy = correct_labels/testX.shape[0]
    #error = 1-accuracy
    conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(testY, y_pred_nn_val)
    #print("Average Error-Neural networks: {}",error)
    print("Confusion Matrix: /n {}".format(conf_matrix))
    print("Accuracy with the test data: {}".format(accuracy))
    print("Per-Class Precision is: {}".format(precision_array))
    print("Per-Class Recall rate: {}".format(recall_array))
	#print('Accuracy of the best model on Test Data(Neural networks) : ', accuracy)
    return (accuracy*100),(max(recall_array)*100),(max(precision_array)*100)