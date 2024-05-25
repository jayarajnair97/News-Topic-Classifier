# News-Topic-Classifier
## News Topic Classification and Clustering
## Project Overview
This project aims to classify and cluster news headlines into predefined topics using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The process involves scraping news headlines from the BBC News website, preprocessing the text data, vectorizing it, applying clustering algorithms, and building classification models. Additionally, a Streamlit app is provided to interactively classify user-provided headlines.

## Technologies and Libraries Used
Web Scraping:
Selenium: For automating web browser interaction.
BeautifulSoup: For parsing HTML and extracting data.

## Natural Language Processing (NLP):
nltk: For text preprocessing, tokenization, stop words removal, and lemmatization.
spaCy: For advanced NLP tasks (not utilized directly in this code but listed for potential use).
gensim: For topic modeling (not utilized directly in this code but listed for potential use).
textblob: For sentiment analysis and other NLP tasks (not utilized directly in this code but listed for potential use).
Machine Learning:
scikit-learn: For text vectorization, clustering (K-means), and classification (Naive Bayes).
tensorflow and keras: For building and training LSTM models.
## Data Visualization and Deployment:
Streamlit: For building the interactive web application.

## Project Steps
Web Scraping:
Use Selenium to navigate to the BBC News website and fetch the page source.
Parse the HTML content with BeautifulSoup to extract news headlines.

## Text Preprocessing:
Remove HTML tags and non-alphanumeric characters.
Tokenize text, remove stop words, and apply lemmatization.
Clean and preprocess the extracted headlines.
TF-IDF Vectorization:
Convert the cleaned headlines into TF-IDF features for further analysis.

## Clustering:
Apply K-means clustering to group the headlines into topics.
Assign human-readable labels to each cluster based on manual inspection.

## Classification:
Train a Naive Bayes classifier using the clustered data.
Evaluate the classifier using accuracy, precision, recall, and F1-score.

## LSTM Model:
Train an LSTM model for headline classification.
Evaluate the LSTM model on a test set.

## Streamlit Application:
Build an interactive web application using Streamlit to allow users to input news headlines and predict their topics.
