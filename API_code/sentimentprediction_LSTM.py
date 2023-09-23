from keras.models import load_model
import re 
import pickle 
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import load_model


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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

embed_dim = 100
units = 64

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model.add(LSTM(units, dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())

adam = optimizers.Adam(learning_rate = 0.001)
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test), verbose=1)

predictions = model.predict(X_test)
y_pred = predictions
matrix_test = metrics.classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1))

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