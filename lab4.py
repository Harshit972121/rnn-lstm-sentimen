# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# 2. Load Dataset
# =========================
data = pd.read_csv(r"C:\Users\Smart-tech\Desktop\Tweets.csv")

# Clean column names
data.columns = data.columns.str.strip()

# =========================
# 3. Handle Missing Values
# =========================
data = data.dropna(subset=['text'])
data['text'] = data['text'].astype(str)

# =========================
# 4. Select Columns
# =========================
texts = data['text']
labels = data['sentiment']

# =========================
# 5. Encode Labels
# =========================
le = LabelEncoder()
y = le.fit_transform(labels)

# =========================
# 6. Text Preprocessing
# =========================
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=100)

# =========================
# 7. Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 8. RNN Model
# =========================
rnn_model = Sequential()
rnn_model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
rnn_model.add(SimpleRNN(64))
rnn_model.add(Dense(3, activation='softmax'))

rnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\nTraining RNN Model...")
rnn_history = rnn_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# =========================
# 9. LSTM Model
# =========================
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
lstm_model.add(LSTM(64))
lstm_model.add(Dense(3, activation='softmax'))

lstm_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\nTraining LSTM Model...")
lstm_history = lstm_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# =========================
# 10. Evaluation
# =========================
print("\nEvaluating Models...")

rnn_loss, rnn_acc = rnn_model.evaluate(X_test, y_test)
lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test)

print("\n===== Final Result =====")
print("RNN Accuracy :", rnn_acc)
print("LSTM Accuracy:", lstm_acc)

# =========================
# 11. Graphs
# =========================

# Accuracy Graph
plt.figure()
plt.plot(rnn_history.history['accuracy'])
plt.plot(lstm_history.history['accuracy'])
plt.plot(rnn_history.history['val_accuracy'])
plt.plot(lstm_history.history['val_accuracy'])
plt.title('Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['RNN Train', 'LSTM Train', 'RNN Val', 'LSTM Val'])
plt.show()

# Loss Graph
plt.figure()
plt.plot(rnn_history.history['loss'])
plt.plot(lstm_history.history['loss'])
plt.plot(rnn_history.history['val_loss'])
plt.plot(lstm_history.history['val_loss'])
plt.title('Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['RNN Train', 'LSTM Train', 'RNN Val', 'LSTM Val'])
plt.show()