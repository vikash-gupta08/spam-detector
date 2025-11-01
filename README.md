             Spam Detector NLP Classification Project

High-Level Design :
Goal: Automatically classify SMS messages as spam or ham (not spam).
Input: Raw SMS text messages with labels (spam/ham).
Output: Predicted label for new SMS messages.
System Components:
Data ingestion and cleaning module
Feature extraction module to convert text into numeric vectors
Classification model training and testing module
Prediction function/module for unseen messages
Evaluation module to measure accuracy, precision, recall
Project Enhancements (Optional)
Add a Flask/FastAPI endpoint to serve predictions via REST.
Build a simple web UI using Streamlit.
Improve accuracy with Logistic Regression or SVM.
Add data visualization (e.g., word cloud of spam messages).


Low-Level Design : 
1. Data Loading and Preprocessing
Load CSV file containing SMS and labels.
Normalize text on these steps:
Convert to lowercase.
Remove punctuation and special characters.
Optionally remove stopwords.
2. Feature Extraction
Use TF-IDF vectorizer or CountVectorizer from sklearn.
Transform SMS text into sparse vector features for modeling.
3. Model Training
Map labels to numeric (ham=0, spam=1).
Split data into train and test sets (e.g., 80:20 ratio).
Train Multinomial Naive Bayes on training data.
Save the trained model using joblib or pickle for reuse.
4. Model Testing and Evaluation
Predict labels for test data.
Calculate accuracy, precision, recall, F1-score.
Print classification report and confusion matrix.
5. Prediction API
Define function to accept new SMS messages.
Apply text preprocessing and vectorization.
Use a saved model to predict labels.
Return spam or ham.
6. Tools
 Pandas, Streamlit, Scikit-learn


   CountVectorizer

             
CountVectorizer is a tool in Python’s scikit-learn library used to convert text documents into a matrix of numerical values—specifically, word (token) counts—so they can be processed by machine learning algorithms.​
How CountVectorizer Works
Bag-of-Words Model:
It tokenizes a collection of texts and builds a vocabulary of unique words. For each document, it creates a vector listing the count of each word from the vocabulary that appears in the document.​
Numeric Representation:
Each row of the matrix represents a document (e.g., an SMS), and each column represents a unique word. The value in each cell is the frequency of that word in the document.​
Example:
If you have texts:
"I like data science"
"Data science is fun"
Vocabulary: [“i”, “like”, “data”, “science”, “is”, “fun”]
Text 1 vector:​
Text 2 vector:​
Features and Options :
Automatic Tokenization: Converts text to lowercase and splits into words.
Stop Word Removal: Removes common words (“the”, “a”) if specified.​
N-gram Support: Can extract bigrams or trigrams for more advanced patterns.
Flexible Preprocessing: Supports custom tokenizers, stemming, lemmatization, and vocabulary limits.​
Use Case :
CountVectorizer is commonly used for basic text feature extraction in tasks such as spam detection, sentiment analysis, and document classification. It prepares text for algorithms that only understand numbers—not raw words.​

Summary :
CountVectorizer transforms plain text into numerical feature vectors, allowing machine learning models to analyze word usage, frequency, and document content effectively for NLP tasks.

                Model Flow Work 


Project Related Data 
CodeBase - https://drive.google.com/drive/folders/1Nn02Su1UGzDZN3T9UovgMVCn7P1L_9BD?usp=sharing
External Spam Data Source - https://drive.google.com/uc?id=1dMXbX9rrZsEIaJrME6EZYBnDl3uYpPrt
Github - https://github.com/vikash-gupta08/spam-detector, uploaded recently

Project Setup on Local Machine
Clone project from Github - https://github.com/vikash-gupta08/spam-detector
Install python
Run pip install pandas scikit-learn streamlit requests
Run streamlit run spamdectormodel.py
Open http://localhost:8501
Project Setup on Local Machine Using Docker 
If you have docker in your local machine then use this cmd
Clone project from Github - https://github.com/vikash-gupta08/spam-detector  
Run docker build -t spam-detector-app .  
Run docker run -p 8501:8501 spam-detector-app
Open http://localhost:8501






# Doc URL
 https://docs.google.com/document/d/13Yq1Rvrz53iRmmdTIOvh9SJdaJPSveNoliy26WzMIX8/edit?tab=t.0
