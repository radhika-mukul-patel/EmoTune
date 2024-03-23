import streamlit as st
import streamlit as st
from fastai.text.all import *
import torch
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import vader
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import pickle
import pathlib

# Set up your Streamlit layout
st.set_page_config(
    page_title='EmoTune',
    layout='centered',
    page_icon="https://logospng.org/download/spotify/logo-spotify-icon-4096.png",
    initial_sidebar_state='auto'
)

if os.name == 'nt':  # Check if running on Windows
    pathlib.Path = pathlib.PosixPath


df = pd.read_parquet("data/lyrics_and_sent.parquet")

@st.cache_data
def download_dicts():
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
    
@st.cache_data
def load_cls_model():
    model_cls = load_learner('models/NLP_model_clasification.pkl')
    return model_cls

def process_text(lyrics):
    # Remove brackets
    lyrics_no_brackets = re.sub(r'\[.*?\]', '', lyrics)
    # Tokenize the lyrics
    word_tokens = word_tokenize(lyrics_no_brackets)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    processed_lyrcs = [w for w in word_tokens if not w in stop_words]
    return processed_lyrcs

def get_sentiment_score(lyrics):
    if isinstance(lyrics, list):
        lyrics = ' '.join(lyrics) # Because tokenized lyrics usually come as a list
    if isinstance(lyrics, str):
        return sid.polarity_scores(lyrics)['compound']
    else:
        st.write(f"Error: Expected a string or a list of words but got {type(lyrics)}")
        return 0

def map_sentiment_type(score):
    if score <= -0.98:
        return 'Very Negative', '#D32F2F'  # Red
    elif score <= -0.6:
        return 'Slightly Negative', '#FF5722'  # Deep Orange
    elif score <= 0.6:
        return 'Neutral', '#9E9E9E'  # Grey
    elif score <= 0.98:
        return 'Slightly Positive', '#4CAF50'  # Green
    else:
        return 'Very Positive', '#2E7D32'  # Dark Green
    
    
def get_similar_songs(df, genre, score):
    # Calculate absolute difference from given score
    df['score_difference'] = abs(df['Sentiment_Score'] - score)

    # Filter by genre
    df_genre = df[df['Genre'] == genre]

    # Separate into two dataframes - one for scores greater than or equal to the given score, and one for scores less than the given score
    df_upper = df_genre[df_genre['Sentiment_Score'] >= score]
    df_lower = df_genre[df_genre['Sentiment_Score'] < score]

    # Sort each dataframe by the absolute difference in score, and take the top 10 rows from each
    upper_songs = df_upper.sort_values(by='score_difference').head(10)
    lower_songs = df_lower.sort_values(by='score_difference').head(10)

    # Concatenate the results
    filtered_songs = pd.concat([upper_songs, lower_songs])

    # Drop the score difference column from the final result
    filtered_songs = filtered_songs.drop(columns='score_difference')
    return filtered_songs


def display_playlist(df):
    st.markdown("## Your Playlist")
    # Iterate over the dataframe and display songs in markdown format
    for idx, row in df.iterrows():
        st.markdown(
            f"<p><span style='font-size:16px;'>{row['Song Name']}</span> - "
            f"<span style='font-size:14px; color:gray;'>{row['Artist']}</span></p>",
            unsafe_allow_html=True
        )


download_dicts()
sid = SentimentIntensityAnalyzer()

#############################################################

bell_icon = '''
<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 16 16">
  <path fill="white" d="M13.377 10.573a7.63 7.63 0 0 1-.383-2.38V6.195a5.115 5.115 0 0 0-1.268-3.446a5.138 5.138 0 0 0-3.242-1.722c-.694-.072-1.4 0-2.07.227c-.67.215-1.28.574-1.794 1.053a4.923 4.923 0 0 0-1.208 1.675a5.067 5.067 0 0 0-.431 2.022v2.2a7.61 7.61 0 0 1-.383 2.37L2 12.343l.479.658h3.505c0 .526.215 1.04.586 1.412c.37.37.885.586 1.412.586c.526 0 1.04-.215 1.411-.586s.587-.886.587-1.412h3.505l.478-.658l-.586-1.77zm-4.69 3.147a.997.997 0 0 1-.705.299a.997.997 0 0 1-.706-.3a.997.997 0 0 1-.3-.705h1.999a.939.939 0 0 1-.287.706zm-5.515-1.71l.371-1.114a8.633 8.633 0 0 0 .443-2.691V6.004c0-.563.12-1.113.347-1.616c.227-.514.55-.969.969-1.34c.419-.382.91-.67 1.436-.837c.538-.18 1.1-.24 1.65-.18a4.147 4.147 0 0 1 2.597 1.4a4.133 4.133 0 0 1 1.004 2.776v2.01c0 .909.144 1.818.443 2.691l.371 1.113h-9.63v-.012z" />
</svg>

'''

timer_icon = '''
<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 14 14"><g transform="translate(14 0) scale(-1 1)"><g fill="none" stroke="white" stroke-linecap="round" stroke-linejoin="round"><path d="M.5 7A6.5 6.5 0 1 0 7 .5a7.23 7.23 0 0 0-5 2"/><path d="m2.5.5l-.5 2L4 3m3 .5v4L4.4 8.8"/></g></g></svg>
'''

settings_icon = '''
<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24"><path fill="white" d="M19.43 12.98c.04-.32.07-.64.07-.98c0-.34-.03-.66-.07-.98l2.11-1.65c.19-.15.24-.42.12-.64l-2-3.46a.5.5 0 0 0-.61-.22l-2.49 1c-.52-.4-1.08-.73-1.69-.98l-.38-2.65A.488.488 0 0 0 14 2h-4c-.25 0-.46.18-.49.42l-.38 2.65c-.61.25-1.17.59-1.69.98l-2.49-1a.566.566 0 0 0-.18-.03c-.17 0-.34.09-.43.25l-2 3.46c-.13.22-.07.49.12.64l2.11 1.65c-.04.32-.07.65-.07.98c0 .33.03.66.07.98l-2.11 1.65c-.19.15-.24.42-.12.64l2 3.46a.5.5 0 0 0 .61.22l2.49-1c.52.4 1.08.73 1.69.98l.38 2.65c.03.24.24.42.49.42h4c.25 0 .46-.18.49-.42l.38-2.65c.61-.25 1.17-.59 1.69-.98l2.49 1c.06.02.12.03.18.03c.17 0 .34-.09.43-.25l2-3.46c.12-.22.07-.49-.12-.64l-2.11-1.65zm-1.98-1.71c.04.31.05.52.05.73c0 .21-.02.43-.05.73l-.14 1.13l.89.7l1.08.84l-.7 1.21l-1.27-.51l-1.04-.42l-.9.68c-.43.32-.84.56-1.25.73l-1.06.43l-.16 1.13l-.2 1.35h-1.4l-.19-1.35l-.16-1.13l-1.06-.43c-.43-.18-.83-.41-1.23-.71l-.91-.7l-1.06.43l-1.27.51l-.7-1.21l1.08-.84l.89-.7l-.14-1.13c-.03-.31-.05-.54-.05-.74s.02-.43.05-.73l.14-1.13l-.89-.7l-1.08-.84l.7-1.21l1.27.51l1.04.42l.9-.68c.43-.32.84-.56 1.25-.73l1.06-.43l.16-1.13l.2-1.35h1.4l.19 1.35l.16 1.13l1.06.43c.43.18.83.41 1.23.71l.91.7l1.06-.43l1.27-.51l.7 1.21l-1.08.84l-.89.7l.14 1.13z"/></svg>
'''

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="https://i.ibb.co/bWTrCwk/emotune.png" alt="EmoTune Logo" style="width: 269px; height: 112px; margin-right: 10px;">
        <div style="display: flex;">
            <span style="margin: 0; margin-left: 250px; margin-right: 25px; color: #1DB954;">{bell_icon}</span>
            <span style="margin: 0; margin-left: auto; margin-right: 25px; color: #1DB954;">{timer_icon}</span>
            <span style="margin: 0; margin-left: auto; margin-right: 25px; color: #1DB954;">{settings_icon}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

st.header("Your Library")

# Add rectangular figures for Music and Podcasts & Shows
st.markdown(
    """
    <div style="display: flex; justify-content: flex-start; margin-top: 10px; margin-bottom: 10px">
        <div style="border: none; background-color: #1DB954; border-radius: 1rem; padding: 0.5rem 1rem; margin-right: 15px; font-size: 20px; color: white;">
            Music
        </div>
        <div style="border: none; background-color: #333333; border-radius: 1rem; padding: 0.5rem 1rem; margin-right: 5px; font-size: 20px;">
            Podcasts & Shows
        </div>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)

# Create a container with a single row
row1, row2, row3, row4 = st.columns(4)

image_url7 = "https://www.hitzound.com/wp-content/uploads/2023/04/Post-Malone-Universal-music.jpg"
album_name1 = "<b style='color: #F2F2F2; text-align: center; font-size: 25px; margin-top: 10px;'>Post Malone</b>"

row1.markdown(
    f"""
    <div style="margin-bottom: 15px;">
        <img src="{image_url7}" alt="image" style="width: 150px; height: 150px; margin-right: 10px;">
            <h3>{album_name1}</h3>
    </div>
    """,
    unsafe_allow_html=True
)


# Add image, name, and description to row1
image_url8 = "https://i.scdn.co/image/ab67706f00000002e25106578cd1e96a32fe9f3b"
album_name2 = "<b style='color: #F2F2F2; text-align: center; font-size: 25px; margin-top: 10px;'>Éxitos España</b>"


row2.markdown(
    f"""
    <div style=" margin-bottom: 15px;">
        <img src="{image_url8}" alt="image" style="width: 150px; height: 150px; margin-right: 10px;">
            <h3>{album_name2}</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Add image, name, and description to row1
image_url9 = "https://i.scdn.co/image/ab67706c0000da8442f7b254aa36658697ce7fea"
album_name3 = "<b style='color: #F2F2F2; text-align: center; font-size: 25px; margin-top: 10px;'>Rap Caviar</b>"


row3.markdown(
    f"""
    <div style=" margin-bottom: 15px;">
        <img src="{image_url9}" alt="image" style="width: 150px; height: 150px; margin-right: 10px;">  
            <h3>{album_name3}</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Add image, name, and description to row1
image_ur20 = "https://images.genius.com/b792656315b4847989b073018533b85d.1000x1000x1.jpg"
album_name4 = "<b style='color: #F2F2F2; text-align: center; font-size: 25px; margin-top: 10px;'>Swaecation</b>"


row4.markdown(
    f"""
    <div style="margin-bottom: 15px;">
        <img src="{image_ur20}" alt="image" style="width: 150px; height: 150px; margin-right: 10px;">
         <h3>{album_name4}</h3>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("<hr>", unsafe_allow_html=True)
st.header("Analyze your lyrics!")

# Create a text input box
lyrics = st.text_area("", placeholder="Enter your lyrics here:")

# Insert custom CSS to center the button
st.markdown("""
    <style>
        .stButton>button {
            display: block;
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

# Create button
analyze_button = st.button("Analyze")

model_cls = load_cls_model()

if lyrics or analyze_button:
    processed_lyrcs = process_text(lyrics)

    # Predict on the clean data
    genre = model_cls.predict(processed_lyrcs)[0]
    st.markdown(f"<h2 style='text-align: center; color: #1DB954;'>Predicted Genre: {genre}</h2>", unsafe_allow_html=True)

    score = get_sentiment_score(processed_lyrcs)
    sentiment_type, color = map_sentiment_type(score)
    # Display the score and type
    st.markdown(f"<h2 style='text-align: center; color: {color};'>Sentiment Score: {score}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: {color};'>Sentiment Type: {sentiment_type}</h2>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)

    # Extract the song details or any other desired information
    playlist = get_similar_songs(df, genre, score)[["Song Name", "Artist"]]
    
    display_playlist(playlist)
