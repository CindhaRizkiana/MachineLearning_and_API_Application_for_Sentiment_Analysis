from keras.models import load_model
import re 
import pickle 
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd
import pickle
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model



df= pd.read_csv('train_preprocess.tsv', sep='\t', header=None)
df.columns =['text', 'label']

def lowercase(text):
    return text.lower()

def remove_unnecessary_char(text):
    text = re.sub('\n',' ',text) # Remove every '\n'
    text = re.sub('rt',' ',text) # Remove every retweet symbol
    text = re.sub('user',' ',text) # Remove every username
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) # Remove every URL
    text = re.sub('  +', ' ', text) # Remove extra spaces
    return text
    
def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) 
    return text

def preprocess(text):
    text = lowercase(text) # 1
    text = remove_nonaplhanumeric(text) # 2
    text = remove_unnecessary_char(text) # 3
    return text

df['text_clean'] = df.text.apply(preprocess)

neg = df.loc[df['label'] == 'negative'].text_clean.tolist()
neu = df.loc[df['label'] == 'neutral'].text_clean.tolist()
pos = df.loc[df['label'] == 'positive'].text_clean.tolist()

neg_label = df.loc[df['label'] == 'negative'].label.tolist()
neu_label = df.loc[df['label'] == 'neutral'].label.tolist()
pos_label = df.loc[df['label'] == 'positive'].label.tolist()

total_data = pos + neu + neg
labels = pos_label + neu_label + neg_label

max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
tokenizer.fit_on_texts(total_data)

X = tokenizer.texts_to_sequences(total_data)

vocab_size = len(tokenizer.word_index)
maxlen = max(len(x) for x in X)

X = pad_sequences(X)
Y = pd.get_dummies(labels)
Y = Y.values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

embed_dim = 100
units = 64

# Create an MLP Classifier model
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # Define the architecture 
    activation='relu',             # Activation function for hidden layers
    solver='adam',                 # Optimizer algorithm
    max_iter=100,                  # Maximum number of iterations
    random_state=42                # Random seed for reproducibility
)

# Train the MLP model on the training data
mlp_classifier.fit(X_train, Y_train)

# Evaluate the model on the test data
accuracy = mlp_classifier.score(X_test, Y_test)
print(f"Test Accuracy: {accuracy}")

# Convert the MLP Classifier to a Keras Sequential model
model = Sequential()
for i, layer in enumerate(mlp_classifier.hidden_layer_sizes, start=1):
    model.add(Dense(layer, activation='relu', input_dim=X_train.shape[1]) if i == 1
              else Dense(layer, activation='relu'))
model.add(Dense(len(np.unique(Y_train)), activation='softmax'))  # Output layer

# Save the Keras model in HDF5 format
model.save("mlp_model.h5")
print(f"Trained MLP model saved to mlp_model.h5")

input_text = """ 
beta sedang berjuang melawan kanker
"""

sentiment = ['negative', 'neutral', 'positive']
model = load_model('mlp_model.h5')


def sentiment_text(input_text, model):
    if(model =="lstm"):
        model = load_model('LSTM.h5')
    elif(model == "RNN"):
        model = load_model('RNN.h5')
    elif(model == "CNN"):
        model = load_model('CNN.h5')
    elif(model == "mlp_model"):
        model = load_model('mlp_model.h5')

    text = [preprocess(input_text)]
    predicted = tokenizer.texts_to_sequences(text)
    guess = pad_sequences(predicted, maxlen=X.shape[1])

    prediction = model.predict(guess)
    print(prediction[0])
    polarity = np.argmax(prediction[0])

    print('Prediction: ',prediction)
    print('Polarity :',polarity)
    print('Text: ',text[0])
    print('Sentiment: ',sentiment[polarity])

    return sentiment[polarity]




def sentiment_file(file, model):
    if(model =="lstm"):
        model = load_model('LSTM.h5')
    elif(model == "RNN"):
        model = load_model('RNN.h5')
    elif(model == "CNN"):
        model = load_model('CNN.h5')
    elif(model == "mlp_model"):
        model = load_model('mlp_model.h5')

    first_column = file.iloc[:, 0]
    file = first_column.astype("string").apply(preprocess)
    print("======== finish preprocess =========")

    file = file.to_frame()
    if(isinstance(file, pd.DataFrame)):
        file.rename(columns={ file.columns[0]: "Tweet" }, inplace = True)
        file["Sentiment"] = None
        file['Tweet'] = file['Tweet'].astype('string')
        file['Sentiment'] = file['Sentiment'].astype('string')

        for i in range(len(file)):
            text = file['Tweet'][i]
            text = [text]

            predicted = tokenizer.texts_to_sequences(text)
            guess = pad_sequences(predicted, maxlen=X.shape[1])

            prediction = model.predict(guess)

            polarity = np.argmax(prediction[0])

            file["Sentiment"][i] =  sentiment[polarity]

        print("======== FINISH TEST =========")
        return file
    else:
        print("======== FAILED TEST =========")
        return "File is Unreadable"