
# coding: utf-8

# In[1]:


# Include so results on different machines are (should be) the same.
import numpy as np
from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)

import random as rn
rn.seed(3)


# In[2]:


get_ipython().system(u'jupyter nbconvert --to script Keras_Character_SimpleRNN.ipynb')


# In[3]:


import glob, os, json, re, unicodedata
from bs4 import BeautifulSoup

load_verbose = 0
loaded_labels = []
loaded_text = []
presidents = [
    "Barack Obama",
    "Donald J. Trump",
#     "Dwight D. Eisenhower",
#     "Franklin D. Roosevelt",
    "George Bush",
#     "George W. Bush",
#     "Gerald R. Ford",
#     "Harry S. Truman",
#     "Herbert Hoover",
#     "Jimmy Carter",
#     "John F. Kennedy",
#     "Lyndon B. Johnson",
#     "Richard Nixon",
#     "Ronald Reagan",
    "William J. Clinton"
]

labels = {}
for idx, name in enumerate(presidents):
    labels[name] = idx

# load raw text files straight in, no parsing
file_to_label = {
    "Obama": "Barack Obama",
    "Trump": "Donald J. Trump",
#     "Eisenhower": "Dwight D. Eisenhower",
#     "Roosevelt": "Franklin D. Roosevelt",
    "Bush": "George Bush",
#     "WBush": "George W. Bush",
#     "Ford": "Gerald R. Ford",
#     "Truman": "Harry S. Truman",
#     "Hoover": "Herbert Hoover",
#     "Carter": "Jimmy Carter",
#     "Kennedy": "John F. Kennedy",
#     "Johnson": "Lyndon B. Johnson",
#     "Nixon": "Richard Nixon",
#     "Reagan": "Ronald Reagan",
    "Clinton": "William J. Clinton"
}

directory = "../data/processed/"
for filename in glob.glob(os.path.join(directory, '*.txt')):
    arr = filename.replace(directory,'').split("_")
    if any(prefix in arr[0] for prefix in file_to_label.keys()):
        loaded_labels = loaded_labels + [labels[file_to_label[arr[0]]]]
        raw = open(filename).read().decode("UTF-8").encode("ascii","ignore")
        loaded_text = loaded_text + [raw] 


print "Loaded", len(loaded_text), "speeches for", len(set(loaded_labels)), "presidents."
# processed2 now contains files generated from unprocessed
directory = "../data/processed3/"
for filename in glob.glob(os.path.join(directory, '*.txt')):
    arr = filename.replace(directory,'').split("_")
    if any(prefix in arr[0] for prefix in file_to_label.keys()):
        loaded_labels = loaded_labels + [labels[file_to_label[arr[0]]]]
        raw = open(filename).read().decode("UTF-8").encode("ascii","ignore")
        loaded_text = loaded_text + [raw] 

print "Loaded", len(loaded_text), "speeches for", len(set(loaded_labels)), "presidents."


# In[4]:


#
# Bagnall 2015 text pre-processing
#
from string import maketrans
import re

chars_to_replace = "[]%!()>=*&_}+"
sub_chars = len(chars_to_replace) * " "
trantab = maketrans(chars_to_replace, sub_chars)
for x in range(0,len(loaded_text)):
    # "Various rare characters that seemed largely equivalent are mapped together..."
    loaded_text[x] = re.sub('`', '\'', loaded_text[x])
    loaded_text[x] = re.sub('--', '-', loaded_text[x])
    loaded_text[x] = re.sub('\n\n', '\n', loaded_text[x])
    # "...all digits in all languages are mapped to 7"
    loaded_text[x] = re.sub('[0-9]+', '7', loaded_text[x])
    # "...any character with a frequency lower than 1 in 10,000 is discarded."
    loaded_text[x] = loaded_text[x].translate(trantab)
    # "Runs of whitespace are collapsed into a single space."
    loaded_text[x] = re.sub(' +', ' ', loaded_text[x])
 
    # REPLACE WORD IN ALL CAPS with <space>; headers
    loaded_text[x] = re.sub('[A-Z]{2,}','', loaded_text[x])

print "Character clean-up complete."


# In[5]:


# Have a look at a scrubbed text excerpt
print loaded_text[200][:750]


# In[6]:


#
# Join all speeches into one massive per president
#  for later processing
#
import numpy as np
from scipy import stats
from operator import itemgetter
from collections import defaultdict

compressed_text = [None]*(len(labels))
for key, value in sorted(labels.iteritems()):
    compressed_text[value] = ""
    for idx in range(0,len(loaded_text)):
        if (loaded_labels[idx] == value):
            compressed_text[value] = compressed_text[value] + loaded_text[idx] + " "
            
print "How many characters of text per president?"
for key, value in sorted(labels.iteritems()):
    print str(value).ljust(2), ":", key.ljust(20), "\t", len(compressed_text[value])

label_min_chars = len(min(compressed_text, key=len))
print "\nMinimum number of characters per president?"
print label_min_chars


# In[7]:


#
# Tokenize words into chars
#
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

# Tokenize into characters
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(compressed_text)
tokenized_text = tokenizer.texts_to_sequences(compressed_text)

# there's an oddity in the encoding for some reason where a len+1 character occurs
unique_chars = len(tokenizer.word_counts)+1

print "Unique char count:", unique_chars
print "\nChars w/ counts:"
print sorted(((v,k) for k,v in tokenizer.word_counts.iteritems()), reverse=True)


# In[8]:


#
# Split speeches into subsequences 
#
from collections import Counter

def splits(_list, _split_size):
    output_list = []
    for idx in range(0, len(_list), _split_size):
        if (idx + _split_size) <= len(_list):
            output_list.append(_list[idx:idx + _split_size])
    return output_list

max_seq_len = 50

# create new speech/label holders
split_text = []
split_labels = []

for idx in range(0, len(tokenized_text)):
    current_label = idx
    current_speech = tokenized_text[idx]#[:label_min_chars]
    current_splits = splits(current_speech, max_seq_len)
    split_text.extend(current_splits)
    split_labels.extend([current_label] * len(current_splits))

print "Subsequence total count; subsequence label total count:", len( split_text ), len( split_labels )
print "\nTotal characters:", len( split_text ) * max_seq_len


# In[9]:


# Have a look at a few split text excerpts
print split_text[10:15]
print split_labels[10:15]


# In[10]:


#
# split amongst speaker samples, not the whole population of samples; shuffle if requested
#
import sklearn.utils

def split_test_train(input_text, input_labels, labels, train_pct=0.8, shuffle_p=True):
    train_text = []
    train_labels = []
    test_text = []
    test_labels = []

    for key, value in sorted(labels.iteritems()):
        # grab all values of a specific label
        subset_text = list(itemgetter(*[idx for idx, label in enumerate(input_labels) if label == value ])(input_text))
        subset_labels = list(itemgetter(*[idx for idx, label in enumerate(input_labels) if label == value ])(input_labels))

        cut_pos = int(train_pct * len(subset_text))
        train_text = train_text + subset_text[:cut_pos]
        train_labels = train_labels + subset_labels[:cut_pos]
        test_text = test_text + subset_text[cut_pos:]
        test_labels = test_labels + subset_labels[cut_pos:]

    if shuffle_p:
        test_text, test_labels = sklearn.utils.shuffle(test_text, test_labels)
        train_text, train_labels = sklearn.utils.shuffle(train_text, train_labels)

    return train_text, train_labels, test_text, test_labels


# In[11]:


#
# Prep test/train
#
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight

# split data smartly
train_X, train_y, test_X, test_y = split_test_train(split_text, split_labels, 
                                                    labels, train_pct=0.8, shuffle_p=False)
print "Sample splits:\n Test = ", len(train_X), "\n Train = ", len(test_X)##

# compute class weights to account for imbalanced classes
y_weights = (class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)).tolist()
y_weights = dict(zip(sorted(labels.values()), y_weights))
print "\nClass weights:\n", y_weights


# In[ ]:


# Have a look at a few of the split text excerpts; 
# likely the same classes based on the split point
print train_X[10:15]
print train_y[10:15]


# In[ ]:


#
# One-hot encoding classes & samples
#
from keras.utils import to_categorical

# one-hot encode classes
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

# one-hot encode samples
train_X = np.array(train_X)
orig_train_X_size=train_X.shape[0]
print "Encoding train_X with dimensions ", train_X.shape
train_X = to_categorical(train_X, num_classes=unique_chars)
print "...to ", train_X.shape
train_X = np.reshape(train_X,(orig_train_X_size,max_seq_len,unique_chars))
print "...and reshaping to ", train_X.shape

test_X = np.array(test_X)
orig_test_X_size=test_X.shape[0]
print "\nEncoding test_X with dimensions ", test_X.shape
test_X = to_categorical(test_X, num_classes=unique_chars)
print "...to ", test_X.shape
test_X = np.reshape(test_X,(orig_test_X_size,max_seq_len,unique_chars))
print "...and reshaping to ", test_X.shape


# In[ ]:


# Have a again look at a few of the split and encoded text excerpts; 
# both arrays should be one-hot encoded.
print train_X[10:11]
print train_y[10:11]


# In[ ]:


##
## BASELINE
##
from keras.layers import Input, Dense, SimpleRNN, Bidirectional, Dropout
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.optimizers import Adagrad, adam
from keras.models import Model
from keras.utils import plot_model

# define operating vars
optimizer='rmsprop'
dropout = 0.5422412690636627
activation = "relu"
batch_size = 50
units = 50
shuffle = True
epochs = 150
merge_mode = 'ave'

# define any callbacks
reduce_lr = ReduceLROnPlateau(monitor='categorical_accuracy', 
                              factor=0.5,
                              patience=1, 
                              verbose=1)
csv_logger = CSVLogger('Keras_Character_SimpleRNN.log')

# assemble & compile model
main_input = Input(shape=(max_seq_len,unique_chars,))
rnn = Bidirectional(SimpleRNN(units=units,
                              activation=activation),
                    merge_mode=merge_mode)(main_input)
drop = Dropout(dropout)(rnn)
main_output = Dense(len(labels),
                    activation='softmax',
                    kernel_initializer='random_normal')(drop)
model = Model(inputs=[main_input], outputs=[main_output])

model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer, 
              metrics=['categorical_accuracy'])
plot_model(model, to_file='Keras_Character_SimpleRNN.png', show_shapes=True, show_layer_names=True)
print(model.summary())


# In[ ]:


# train the model
model.fit([np.array(train_X)],
          [np.array(train_y)],
          batch_size=batch_size,
          epochs=epochs,
          shuffle=shuffle,
          class_weight = y_weights,
          callbacks=[reduce_lr, csv_logger],
          verbose=1)

model.save('Keras_Character_SimpleRNN.h5')
print ("Model saved.")
del model


# In[ ]:


##
## MODEL EVALUATION
##


# In[ ]:


# Load computed model
from keras.models import load_model
# returns a compiled model identical to the one trained
model = load_model('Keras_Character_SimpleRNN.h5')
print ("Model re-loaded.")


# In[ ]:


from sklearn import metrics

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
print "\n\nDone prediction."

print "\nAUC = ", metrics.roc_auc_score(test_y, pred_y)


# In[ ]:


# Plot confusion matrix
#   from scikit-learn examples @
#   http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html 
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

plt.show()


# In[ ]:


#Plot normalized confusion matrix
plt.figure(figsize=(10,10))
plot_confusion_matrix(np.round(cnf_matrix,2), classes=(sorted(labels, key=labels.get)), normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[ ]:


count = 0
def reverse_map(map):
    return dict((v,k) for k,v in map.iteritems())
token_word_map = reverse_map(tokenizer.word_index)
president_map = reverse_map(labels)
def to_words(a):
    return " ".join([token_word_map[id] for id in a if id != 0])
sample_size = 20
def print_row(row):
    print "".join(col.ljust(22) for col in row)
print_row(["Predicted","Correct","Sentence"])
print_row(["----------","----------","---------------"])
sample = []
for i in range(len(test_y_collapsed)):
    if (pred_y_collapsed[i] != test_y_collapsed[i] and count < sample_size):
        sample += [[president_map[pred_y_collapsed[i]], president_map[test_y_collapsed[i]], to_words(test_X[i])]]
        count += 1
        
for row in sample:
    print_row(row)


# In[ ]:


##
## MODEL OPTIMIZATION
##
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# define operating vars
#optimizer='rmsprop'
#dropout = 0.5422412690636627
#activation = "relu"
#batch_size = 50
#units = 50
#shuffle = True
#epochs = 150
#merge_mode = 'ave'

def create_model(optimizer='rmsprop', learn_rate=0.01,
                 init_mode11='glorot_uniform', init_mode12='glorot_uniform', 
                 merge_mode='ave', activation='relu', 
                 dropout_rate=0.0, neuron_count=50):
    # assemble & compile model
    main_input = Input(shape=(max_seq_len,unique_chars,))
    rnn = Bidirectional(SimpleRNN(neuron_count=units,
                                  activation=activation,
                                  kernel_initializer=init_mode1),
                        merge_mode=merge_mode)(main_input)
    drop = Dropout(dropout)(rnn)
    main_output = Dense(len(labels),
                        activation='softmax',
                        kernel_initializer=init_mode2)(drop)
    return Model(inputs=[main_input], outputs=[main_output])
 
# create model
model = KerasClassifier(build_fn=create_model, verbose=1, epochs=5)

# define the grid search parameters
epoch = [5]
batch_sizes = [25, 50, 75, 100, 200]
optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
learn_rates = [0.001, 0.01, 0.1, 0.2, 0.3]
init_modes1 = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
init_modes2 = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
merge_modes = ['ave','min','max','add']
activations = ['relu','sigmoid','tanh']
dropout_rates = [0.0,0.2,0.4,0.6,0.8]
neuron_count = [25,50,75,100,150,200]
param_grid = dict(batch_size=batch_sizes,
                  epochs=epoch,
                  optimizer=optimizers,
                  learn_rate=learn_rates,
                  init_mode1=init_modes1,
                  init_mode2=init_modes2,
                  merge_mode=merge_modes,
                  activation=activations,
                  dropout_rate=dropout_rates,
                  neurons=neuron_count)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)

# summarize results
# from http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

