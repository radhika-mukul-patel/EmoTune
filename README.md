![EmoTune](https://i.ibb.co/bWTrCwk/emotune.png)

Generate Spotify playlists based on song lyrics' sentiment and genre.

---------------
https://emotune.streamlit.app/
--------------

## Project Summary

### Goal Definition

EmoTune's aim is to create music playlists derived from the sentiment analysis and genre classification of song lyrics. This required the assembly of a repository containing song lyrics data, the development of two distinct models (one for sentiment analysis, another for genre classification), and an application framework for a user-friendly interface.

### Data Sourcing

The Genius API, a well-curated lyrics finder boasting over 1.7 million songs, was the primary data source. Lyrics from 1000 songs across 10 genres and 50 different artists were gathered and compiled into a single CSV file.

### The Models

#### Sentiment Analysis

Models explored for sentiment analysis included Flair, Textblob, Hugging Face (amanda-cristina/fine tuned on lyrics), and VADER. We adopted VADER due to its ability to give an overall score for larger texts. After classifying sentiment scores into five categories (from very negative to very positive) with VADER, we assigned these categories to each song based on their sentiment score.

#### Genre Classification

Our approach for genre classification was threefold. First, we attempted tf-idf vectorization with ML models such as Naive Bayes, SVM, Max Entropy, and Random Forest. Second, we utilized the fast.ai library and applied transfer learning techniques with an AWD-LSTM model. Finally, we explored the roBERTa model via the SimpleTransformers library. We opted for AWD-LSTM, a type of recurrent neural network architecture designed to remember long-term dependencies in sequence data, which led to a final accuracy of around 0.56.

### Streamlit Application

Streamlit was our choice for creating the application's UI. As it requires no specific setup, prerequisites or configurations from the user, it offers a quick and easy-to-use interface.

![image](https://github.com/felipebasurto/EmoTune/assets/62935664/5208fbc1-2287-478e-aa42-8522677844a1)
![image](https://github.com/felipebasurto/EmoTune/assets/62935664/4b0878af-7f70-4db4-93c1-d5a1276d734b)
![image](https://github.com/felipebasurto/EmoTune/assets/62935664/d4c595a4-2777-48bc-9081-0f8e1de6f7ce)

## User Manual
(TO-DO)

