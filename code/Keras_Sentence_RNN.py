
# coding: utf-8

# In[1]:

#get_ipython().system(u'jupyter nbconvert --to script Keras_Sentence_RNN.ipynb')
import preprocessing
import numpy as np


# In[2]:

import collections

sentence_max_threshold = 50000
tokenizer, max_sentence_len_word, labels, train_X, test_X, train_y, test_y = preprocessing.get_data(sentence_max_threshold)
print train_X.shape, train_y.shape, len(tokenizer.word_counts)#, len(tokenized_text)

x_count = collections.Counter()
for i in range(len(test_y)):
    x_count.update({str(test_y[i]): 1})

for key, value in sorted(x_count.iteritems(), reverse=True):
    print key, value, float(value)/sentence_max_threshold


# ### Use Keras_Sentence_RNN.py to avoid time-out problem
# If the trained model runs too long, it will time out. To get around this issue, you can skip the run here and instead use Keras_Sentence_RNN.py to train and save the model, then load the saved model here to predict the data.
# 

# In[5]:

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, SimpleRNN, Dropout

max_features = len(tokenizer.word_counts) + 1 # + 1 is for padded word id 0s
batch_size = 100
num_of_nodes = 100 #max_sentence_len_word

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, num_of_nodes, input_length=max_sentence_len_word))
model.add(SimpleRNN(num_of_nodes,activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(len(labels), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adagrad',metrics=['categorical_accuracy'])
print(model.summary())

model.fit(train_X, y=train_y, batch_size=batch_size, nb_epoch=20, verbose=1, class_weight='auto')


# In[ ]:

from keras.models import load_model

# Load the model you trained Keras_Sentence_RNN.py here to predict on test data
model.save('rnn.h5')  
model = load_model('rnn.h5')


# In[ ]:

# Evaluate performance
print "Evaluating test data..."
loss_and_metrics = model.evaluate(test_X, test_y)
print model.metrics_names
print loss_and_metrics

# Make some predictions
print "\nPredicting using test data..."
pred_y = model.predict(test_X, batch_size=batch_size, verbose=1)
pred_y_collapsed = np.argmax(pred_y, axis=1)
test_y_collapsed = np.argmax(test_y, axis=1)

print "Done prediction."

from sklearn import metrics
print "\nAUC = ", metrics.roc_auc_score(test_y, pred_y)


# In[ ]:

# from scikit-learn examples @
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html 
get_ipython().magic(u'matplotlib inline')
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_y_collapsed, pred_y_collapsed)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(10,10))
plot_confusion_matrix(cnf_matrix, classes=(sorted(labels, key=labels.get)),
                      title='Confusion matrix, without normalization')

# #Plot normalized confusion matrix
# plt.figure(figsize=(10,10))
# plot_confusion_matrix(cnf_matrix, classes=(sorted(labels, key=labels.get)), normalize=True,
#                       title='Normalized confusion matrix')

plt.show()


# In[ ]:



