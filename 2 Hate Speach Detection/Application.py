# Importing Modules
import re
import nltk
import pickle as pkl
import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st

nltk.download("stopwords")
nltk.download("wordnet")

vectorizer = pkl.load(open("Material/vectorizer.sav", "rb"))
model = pkl.load(open("Material/finalized_model.sav", "rb"))
df = pd.read_csv("Material/cleaned_data.csv")

# Navigation Bar
rad = st.sidebar.radio("Web Pages", ("Home", "Detect Hate"))

# Cleaning Functions
# Function to convert abbrevations into origional word
def clean_abbreviation(token):
    if token == 'u':
        return 'you'
    if token == 'r':
        return 'are'
    if token == 'some1':
        return 'someone'
    if token == 'yrs':
        return 'years'
    if token == 'hrs':
        return 'hours'
    if token == 'mins':
        return 'minutes'
    if token == 'secs':
        return 'seconds'
    if token == 'pls' or token == 'plz':
        return 'please'
    if token == '2morow':
        return 'tomorrow'
    if token == '2day':
        return 'today'
    if token == "shouldn't":
        return 'should not'
    if token == '4got' or token == '4gotten':
        return 'forget'
    if token == 'amp' or token == 'quot' or token == 'lt' or token == 'gt' or token == '½25':
        return ''
    return token

# Function to clean the data
def clean_data(txt):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    STOPWORDS = list(nltk.corpus.stopwords.words("english"))
    txt = txt.lower()                             # lowering text
    txt = re.sub(r'#', '', txt)                   # Removing hashtags
    txt = re.sub(r'@[A-Za-z0-9]+', '', txt)       # Removing Mentions
    txt = re.sub(r'https?:\/\/\S+', '', txt)      # Removing Links
    txt = re.sub(r'rt[\s]+', '', txt)             # Removing Retweets
    txt = re.sub(r'\n', ' ', txt)                 # Removing Newline
    txt = re.sub(r"[^a-zA-Z0-9]"," ", txt)        # Removing all special characters
    txt = " ".join([clean_abbreviation(i) for i in txt.split()])                           # Checking for abbreviations
    txt = " ".join([lemmatizer.lemmatize(i) for i in txt.split() if i not in STOPWORDS])   # Removing stopwrods and applying lemmatization
    return txt


def detect_speech():
    res = None
    st.title("Hate Speech Recognition")
    with st.form("detect_form", clear_on_submit=False):
        data = st.text_area("Put your Speech here", height=250)
        submitted = st.form_submit_button("Detect")
        if submitted:
            data = str(data)
            data = clean_data(data)
            df.loc[0, "clean_tweet"] = data
            print(df.loc[0, "clean_tweet"])
            data = vectorizer.fit_transform(df.clean_tweet)
            data = data[:10]
            res = model.predict(data)
            res = res[0]

            st.write("Processing......")
            st.write("Predicted result is")
            st.subheader(f"{str(res).upper()}")

if rad == "Home":
    st.title("Hate Speech Recognition")
    st.write("Hate Speech Detection is the automated task of detecting if a piece of text contains hate speech. The detection of hate speech in social media is a crucial task. The uncontrolled spread of hate has the potential to gravely damage our society, and severely harm marginalized people or groups. A major arena for spreading hate speech online is social media.")
    st.write("In this project I built a system that can detect if a speech is Offensive, Hate or Normal. And for that I used supervised machine learning and trained multiple machine leaning algorithms to find the best one and then that model used here in this GUI to make predictions. ")
    st.header("Dataset")
    st.write("Dataset that I used here is an open source data and has been extracted from Twitter. Dataset using Twitter data, was used to research hate-speech detection. The text is classified as: hate-speech, offensive language, and neither. Due to the nature of the study, it’s important to note that this dataset contains text that can be considered racist, sexist, homophobic, or generally offensive.")
    st.header("Exploration")
    st.write("Insights that I get from the exploration of this data are given below.")
    img = cv.imread("Material/Img1.png")
    st.image(img)
    st.write("Here I checked the distribution of data among the classes in the target feature and you can see that most of the records are for Offensive. And for others less number of records are present that's mean the data is imbalance that is why I used Oversampling technique to balance the data.")
    img = cv.imread("Material/Img2.png")
    st.image(img)
    st.write("Above graph show the most common words used in Hate Speeches")
    st.text("Go to the Detection Page to check if your speech is Hated, Offensive or Normal")

if rad == "Detect Hate":
    detect_speech()
