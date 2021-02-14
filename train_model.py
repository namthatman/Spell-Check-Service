# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Install Libraries
#!pip install fairseq
#!pip install transformers -U
#!pip install vncorenlp


# Import Libraries
import tensorflow as tf
import tensorflow.keras.backend as K
from transformers import *
from fairseq.data import Dictionary
from sklearn.model_selection import train_test_split
#from vncorenlp import VnCoreNLP

# Clear keras session
tf.keras.backend.clear_session()


# GPU
tf.debugging.set_log_device_placement(True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)


# TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)
AUTO = tf.data.experimental.AUTOTUNE



## CONTINUE WORK

# Load Dataset
train_set = pd.read_csv("resources/train_dataset/train_f.csv")
validate_set = pd.read_csv("resources/train_dataset/validate_f.csv")
test_set = pd.read_csv("resources/train_dataset/test_f.csv")
X_train = train_set['0']
y_train = train_set['label']
X_validate = validate_set['0']
y_validate = validate_set['label']
y_validate = np.asarray(y_validate)
X_test = test_set['0']
y_test = test_set['label']
y_test = np.asarray(y_test)

# Load the dictionary  
vocab = Dictionary()
#vocab.add_from_file("../input/phobert-base-transformers/dict.txt")
vocab.add_from_file("PhoBERT_base_transformers/new_vocab.txt")

# Tokenizing Train, Validation, Test
MAX_LEN = 96
#MAX_LEN = 256
ctrain = X_train.shape[0]
cval = X_validate.shape[0]
ctest = X_test.shape[0]


# Train input
input_ids_train = np.ones((ctrain,MAX_LEN),dtype='int32')
attention_mask_train = np.zeros((ctrain,MAX_LEN),dtype='int32')
token_type_ids_train = np.zeros((ctrain,MAX_LEN),dtype='int32')


for k in range(ctrain):
    
    text = "<s> " + str(X_train.loc[k]) + " </s>"
    enc = vocab.encode_line(text, append_eos=False, add_if_not_exist=True).long().tolist()
    input_ids_train[k,:len(enc)] = enc
    attention_mask_train[k,:len(enc)] = 1


# Validate input
input_ids_val = np.ones((cval,MAX_LEN),dtype='int32')
attention_mask_val = np.zeros((cval,MAX_LEN),dtype='int32')
token_type_ids_val = np.zeros((cval,MAX_LEN),dtype='int32')

for k in range(cval):
    
    text = "<s> " + str(X_validate.loc[k]) + " </s>"
    enc = vocab.encode_line(text, append_eos=False, add_if_not_exist=True).long().tolist()
    input_ids_val[k,:len(enc)] = enc
    attention_mask_val[k,:len(enc)] = 1


# Test input
input_ids_test = np.ones((ctest,MAX_LEN),dtype='int32')
attention_mask_test = np.zeros((ctest,MAX_LEN),dtype='int32')
token_type_ids_test = np.zeros((ctest,MAX_LEN),dtype='int32')

for k in range(ctest):
    
    text = "<s> " + str(X_test.loc[k]) + " </s>"
    enc = vocab.encode_line(text, append_eos=False, add_if_not_exist=True).long().tolist()
    input_ids_test[k,:len(enc)] = enc
    attention_mask_test[k,:len(enc)] = 1

    
    
##




## START FROM SCRATCH

# Load Dataset
#false_articles = pd.read_csv("../input/spellingcheck/false_articles.csv", header=None)
false_articles = pd.read_csv("train_dataset/false_articles_full.csv", header=None)
true_articles = pd.read_csv("train_dataset/true_articles.csv", header=None)


# Load the dictionary  
vocab = Dictionary()
#vocab.add_from_file("../input/phobert-base-transformers/dict.txt")
vocab.add_from_file("PhoBERT_base_transformers/new_vocab.txt")


# Data Preprocessing
true_articles['label'] = 1
true_articles.head()
false_articles['label'] = 0
false_articles.head()

all_articles = pd.concat([true_articles, false_articles], ignore_index=True)
#all_articles.head()
#all_articles.tail()


drop_idx = []
for i in range(all_articles.shape[0]):
    text = str(all_articles.loc[i,0]).split()
    if len(text)>93 or len(text)<3:
        drop_idx.append(all_articles.index[i])
        
len(drop_idx)
all_articles.shape[0]
all_articles = all_articles.drop(drop_idx).reset_index(drop=True)
all_articles.shape[0]




# Train, Validation, Test splits
test_set = all_articles.sample(n=100, random_state=258).reset_index(drop=True)
X_test = test_set[0]
y_test = test_set['label']
all_articles = all_articles.drop(test_set.index).reset_index(drop=True)
X_train, X_validate, y_train, y_validate = train_test_split(all_articles[0], all_articles['label'], test_size=0.1, random_state=258)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_validate = X_validate.reset_index(drop=True)
y_validate = y_validate.reset_index(drop=True)


# Tokenizing Train, Validation, Test
MAX_LEN = 96
#MAX_LEN = 256
ctrain = X_train.shape[0]
cval = X_validate.shape[0]
ctest = X_test.shape[0]
ctrain
cval
ctest


# Train input
input_ids_train = np.ones((ctrain,MAX_LEN),dtype='int32')
attention_mask_train = np.zeros((ctrain,MAX_LEN),dtype='int32')
token_type_ids_train = np.zeros((ctrain,MAX_LEN),dtype='int32')


for k in range(ctrain):
    
    text = "<s> " + str(X_train.loc[k]) + " </s>"
    enc = vocab.encode_line(text, append_eos=False, add_if_not_exist=True).long().tolist()
    input_ids_train[k,:len(enc)] = enc
    attention_mask_train[k,:len(enc)] = 1


# Validate input
input_ids_val = np.ones((cval,MAX_LEN),dtype='int32')
attention_mask_val = np.zeros((cval,MAX_LEN),dtype='int32')
token_type_ids_val = np.zeros((cval,MAX_LEN),dtype='int32')

for k in range(cval):
    
    text = "<s> " + str(X_validate.loc[k]) + " </s>"
    enc = vocab.encode_line(text, append_eos=False, add_if_not_exist=True).long().tolist()
    input_ids_val[k,:len(enc)] = enc
    attention_mask_val[k,:len(enc)] = 1


# Test input
input_ids_test = np.ones((ctest,MAX_LEN),dtype='int32')
attention_mask_test = np.zeros((ctest,MAX_LEN),dtype='int32')
token_type_ids_test = np.zeros((ctest,MAX_LEN),dtype='int32')

for k in range(ctest):
    
    text = "<s> " + str(X_test.loc[k]) + " </s>"
    enc = vocab.encode_line(text, append_eos=False, add_if_not_exist=True).long().tolist()
    input_ids_test[k,:len(enc)] = enc
    attention_mask_test[k,:len(enc)] = 1
    


# Save Train, Validate, Test set, Vocab
train_set = pd.merge(X_train, y_train, left_index=True, right_index=True)
validate_set = pd.merge(X_validate, y_validate, left_index=True, right_index=True)
test_set = pd.merge(X_test, y_test, left_index=True, right_index=True)

train_set.to_csv('train_f.csv', index=False, header=True)
validate_set.to_csv('validate_f.csv', index=False, header=True)
test_set.to_csv('test_f.csv', index=False, header=True)


#f = open("new_vocab.txt","w+")
#f.close() 
#vocab.save("./new_vocab.txt")


##




# Build PhoBERT model
def build_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
    config = RobertaConfig.from_pretrained("PhoBERT_base_transformers/config.json")
    phobert = TFRobertaModel.from_pretrained("PhoBERT_base_transformers/model.bin", from_pt=True, config=config)
    
    x = phobert([ids, att, tok])
    
    last_hidden_states = x[0]
    
    clf_output = last_hidden_states[:, 0, :]
    
    x1 = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    x1 = tf.keras.layers.Dropout(0.2)(x1)
    x1 = tf.keras.layers.Dense(32, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.2)(x1)
    out = tf.keras.layers.Dense(2, activation='softmax')(x1)
    
    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=out)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) # lr=3e-5
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    return model


# Train    
model = build_model()


# TPU train
with strategy.scope():
    model = build_model()
        
model.summary()


# Load weights
#model.load_weights("../input/weights/tpu_phobert.h5")
model.load_weights("resources/saved_model/gpu_phobert_v3_best.h5")

#optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
#model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])



# Fit model
sv = tf.keras.callbacks.ModelCheckpoint(
        'resources/saved_model/gpu_phobert_v3_best_new.h5', monitor='val_accuracy', verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch')


model.fit([input_ids_train, attention_mask_train, token_type_ids_train], y_train, 
          epochs=1, batch_size=32, verbose=1, callbacks=[sv], 
          validation_split=0.1) # batch=32


# Save model
model.save_weights("resources/saved_model/gpu_phobert_v3_best_new.h5")
#model.save("./tpu_phobert_f.h5")
model.save('resources/saved_model/gpu_phobert_new')


# Load entire model
model = tf.keras.models.load_model("resources/saved_model/gpu_phobert")


# Evaluate model
eval_score = model.evaluate([input_ids_val,attention_mask_val,token_type_ids_val], y_validate, verbose=1, batch_size=32)
eval_score


# Predict Test
y_pred = model.predict([input_ids_test,attention_mask_test,token_type_ids_test], verbose=1, batch_size=32)
y_pred_label = np.argmax(y_pred, axis=-1)
test_set['prediction'] = y_pred_label
test_set


# Evaluate Test
test_score = model.evaluate([input_ids_test,attention_mask_test,token_type_ids_test], y_test, verbose=1, batch_size=32)


# Single test input
test_input = ["xuất sắc", "suất sắc", "xuất xắc", "suất xắc", "anh yêu em", "anh yêu yêu em", "anh tìm nỗi nhớ", "anh tìm nổi nhớ", 
              "đua chen dày vò xâu xé quanh thân xác nát nhàu", "đua chenh dài vò sâu sé quanh thân xác nát nhàu",
             "Trước tình hình trên, các biện pháp phòng, chống dịch đã được thành phố Đà Nẵng triển khai khẩn cấp", 
             "Trước tình hình trên, các biện pháp phòng, trông dịch đã được thành phố đà Nẵng triển khai khẩnn cấ"]
cs = len(test_input)

input_ids_s = np.ones((cs,MAX_LEN),dtype='int32')
attention_mask_s = np.zeros((cs,MAX_LEN),dtype='int32')
token_type_ids_s = np.zeros((cs,MAX_LEN),dtype='int32')

for k in range(cs):
    
    text = "<s> " + str(test_input[k]) + " </s>"
    enc = vocab.encode_line(text, append_eos=False, add_if_not_exist=False).long().tolist()
    input_ids_s[k,:len(enc)] = enc
    attention_mask_s[k,:len(enc)] = 1
    
s_pred = model.predict([input_ids_s,attention_mask_s,token_type_ids_s], verbose=1)
s_pred_label = np.argmax(s_pred, axis=-1)
s_pred_label


# Display with labels
single_case = pd.DataFrame(test_input)
single_case['prediction'] = s_pred_label
single_case


# Display with probabilities
single_case = pd.DataFrame(test_input)
single_case_0 = pd.DataFrame(s_pred[:,0])
single_case_1 = pd.DataFrame(s_pred[:,1])
single_case['p0'] = single_case_0
single_case['p1'] = single_case_1
single_case