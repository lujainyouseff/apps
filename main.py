import streamlit as st
from transformers import pipeline
import cleantext
import pandas as pd

# CSS for background images and custom styles
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {   
  background-image: url("https://i.pinimg.com/originals/e5/1d/7f/e51d7f568c7234b292427e817501d8f2.jpg");
  background-size: contain;
  background-repeat: no-repeat;
  background-position: top right;
  background-color: black;
}
[data-testid="stHeader"] {
  background-color: rgba(0,0,0,0);
}
[data-testid="stToolbar"] {
  right: 2rem;
}
[data-testid="stSidebar"]::before {
  content: "";
  background-image: url("https://i.ibb.co/XDdDb2g/ezgif-com-gif-maker-1.gif");
  background-size: 100%;
  background-repeat: no-repeat;
  background-position: center bottom;
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  opacity: 0.5;
  z-index: 0;
}
[data-testid="stSidebar"]::after {
  content: "";
  background-color: #EEEEEE;
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: -1;
}
[data-testid="stExpander"] {
  background: url("https://i.pinimg.com/originals/e5/1d/7f/e51d7f568c7234b292427e817501d8f2.jpg");
  background-size: auto 150%;
  background-repeat: no-repeat;
  background-position: top right;
  background-color: black;
}
#custom-header {
  color: #C73659; 
  font-size: 20px; 
}
#custom-header_2 {
  color: #C73659; 
  font-size: 20px; 
}

#custom-header_3 {
  color: #C73659; 
  font-size: 20px; 
}
</style>
"""
# Inject CSS with markdown
st.markdown(page_bg_img, unsafe_allow_html=True)
# Streamlit headers
st.header('Reviews Sentiment Analysis ')
st.sidebar.markdown('<h2 id="custom-header">Lujain Yousef. </h2>', unsafe_allow_html=True)
st.sidebar.markdown('<h2 id="custom-header_2">Data Science Student. </h2>', unsafe_allow_html=True)
st.sidebar.markdown('<h2 id="custom-header_3">Sentiment Analysis Project. </h2>', unsafe_allow_html=True)
# First expander for text analysis
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        # Create the pipeline for DistilBERT
        distil_pipeline = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        distil_output = distil_pipeline(text)

        # Create the pipeline for BERT with from_pt=True
        bert_pipeline = pipeline(task="sentiment-analysis", model="kwang123/bert-sentiment-analysis")
        bert_output = bert_pipeline(text)

        # Display the results
        st.write('DistilBERT Output: ', round(distil_output[0]['score'], 3),'DistilBERT Label:',distil_output[0]['label'])
        st.write('BERT Output: ', round(bert_output[0]['score'], 3),'BERT Label: ',bert_output[0]['label'])

    # Text input for cleaning text
    pre = st.text_input('Clean Text: ')
    if pre:
        cleaned_text = cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                       stopwords=True, lowercase=True, numbers=True, punct=True)
        st.write('Cleaned Text: ', cleaned_text)

        distil_pipeline = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        distil_output = distil_pipeline(cleaned_text)

        # Create the pipeline for BERT with from_pt=True
        bert_pipeline = pipeline(task="sentiment-analysis", model="kwang123/bert-sentiment-analysis")
        bert_output = bert_pipeline(cleaned_text)

        st.write('DistilBERT Output: ', round(distil_output[0]['score'], 3), 'DistilBERT Label:',
                 distil_output[0]['label'])
        st.write('BERT Output: ', round(bert_output[0]['score'], 3), 'BERT Label: ', bert_output[0]['label'])


@st.cache_resource
def load_sentiment_pipeline():
    sentiment_pipeline_multilingual = pipeline(task="sentiment-analysis", model='nlptown/bert-base-multilingual-uncased-sentiment')
    return  sentiment_pipeline_multilingual

# Cache the text cleaning function
@st.cache_data
def clean_text(text):
    return cleantext.clean(text, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True)

# Function to get sentiment score
def get_score(text, sentiment_pipeline_multilingual):
    cleaned_text = clean_text(text)
    result = sentiment_pipeline_multilingual(cleaned_text)
    return result[0]['score']

# Function to get sentiment label
def get_label(text, sentiment_pipeline):
    cleaned_text = clean_text(text)
    result = sentiment_pipeline(cleaned_text)

    # Map the original labels to the new labels
    label_mapping = {
        '1 star':  '⭐',
        '2 stars': '⭐⭐',
        '3 stars': '⭐⭐⭐',
        '4 stars': '⭐⭐⭐⭐',
        '5 stars': '⭐⭐⭐⭐⭐'
    }

    result[0]['label'] = label_mapping.get(result[0]['label'], result[0]['label'])
    return result[0]['label']

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file', type=['csv'])

    if upl is not None:
        df = pd.read_csv(upl)

        # Assuming the CSV has a column named 'Reviews' for analysis
        if 'Reviews' in df.columns:
            sentiment_pipeline_multilingual = load_sentiment_pipeline()

            df['score'] = df['Reviews'].apply(lambda x: get_score(x, sentiment_pipeline_multilingual))
            df['label'] = df['Reviews'].apply(lambda x: get_label(x, sentiment_pipeline_multilingual))

            st.write(df.head(100))

            # Function to convert dataframe to CSV
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(df)

            st.download_button(
                label='Download data as CSV',
                data=csv,
                file_name='sentiment.csv',
                mime='text/csv',
            )
        else:
            st.error("The uploaded CSV does not contain a 'Reviews' column.")



