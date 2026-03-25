# 🧠 RNN vs LSTM Sentiment Analysis

This project performs **Sentiment Analysis on Tweets** using Deep Learning models:

* Simple RNN
* LSTM (Long Short-Term Memory)

The goal is to compare the performance of both models.

---

## 📂 Dataset

* Dataset used: `Tweets.csv`
* Contains tweet text and sentiment labels (positive, negative, neutral)

---

## ⚙️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib
* Scikit-learn
* TensorFlow / Keras

---

## 🔄 Workflow

1. Load Dataset
2. Data Cleaning (remove missing values)
3. Text Preprocessing

   * Tokenization
   * Padding
4. Label Encoding
5. Train-Test Split
6. Build Models

   * RNN
   * LSTM
7. Model Training
8. Evaluation
9. Visualization (Accuracy & Loss Graphs)

---

## 🤖 Models Used

### 🔹 RNN (SimpleRNN)

* Basic sequential model
* Works well for short dependencies

### 🔹 LSTM

* Advanced version of RNN
* Handles long-term dependencies
* Usually gives better accuracy

---

## 📊 Results

* Both models are trained and evaluated on test data
* Accuracy and Loss graphs are plotted
* LSTM generally performs better than RNN

---

## ▶️ How to Run

1. Open Jupyter Notebook
2. Run all cells step-by-step
3. Make sure dataset path is correct:

   ```
   C:\Users\Smart-tech\Desktop\Tweets.csv
   ```

---

## 📌 Output

* Model Accuracy
* Model Loss
* Graphs comparing RNN vs LSTM

---

## 📎 File Included

* `RNN_LSTM_Sentiment_Analysis.ipynb`

---

## 👨‍💻 Author

Harshit Patel

---

## ⭐ Note

This project is useful for:

* Beginners in Deep Learning
* Understanding RNN vs LSTM
* NLP (Natural Language Processing) basics
