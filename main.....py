import streamlit as st
from transformers import pipeline
import cleantext
import pandas as pd
import requests
from PIL import Image
from io import BytesIO


page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {   
  background-image: url("https://i.ibb.co/VqMHJ0r/Untitled-design-27.png");
  background-size: auto 50%;
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
  background-color: #EEEEEE;
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
  background-color: #000000;
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: -1;
}

[data-testid="stExpander"] {
  background: url("https://i.ibb.co/VqMHJ0r/Untitled-design-27.png");
  background-size: auto 50%;
  background-repeat: no-repeat;
  background-position: top right;
  background-color: black;
}



[data-testid="stExpander"].st-expander-content {
  opacity: 0.5;
}

/* Text input boxes and file upload buttons */
[data-testid="stTextInput"] {
  background-color: rgba(0, 0, 0, 0);  
  color: white;  
  border: 1px solid rgba(0, 0, 0, 0);  /* Border with full transparency */
  backdrop-filter: blur(2px);  /* Blur effect */
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  /* Box shadow */
  padding: 10px;
  border-radius: 8px;
}

[data-testid="stFileUploader"] div[role="button"] {
  background-color: rgba(0, 0, 0, 0);  /* Fully transparent background */
  color: white;  /* Text color */
  border: 1px solid rgba(0, 0, 0, 0);  /* Border with full transparency */
  backdrop-filter: blur(2px);  /* Blur effect */
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  /* Box shadow */
  padding: 10px;
  border-radius: 8px;
}

/* Hover effects for text input and file upload buttons */
[data-testid="stTextInput"]:hover,
[data-testid="stFileUploader"] div[role="button"]:hover {
  background-color: rgba(255, 255, 255, 0.05);  /* Slightly less transparent on hover */
}

/* Focus effects for text input and file upload buttons */
[data-testid="stTextInput"]:focus,
[data-testid="stFileUploader"] div[role="button"]:focus {
  outline: none;
  box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);  /* Highlight on focus */
}

#custom-header {
  color: white;
  font-size: 20px; 
}

#custom-header_2 {
  color: white; 
  font-size: 20px; 
}

#custom-header_3 {
  color: white; 
  font-size: 20px; 
}

#custom-header_5 {
  color: white; 
  font-size: 20px; 
}
</style>
"""


# Inject CSS with markdown
st.markdown(page_bg_img, unsafe_allow_html=True)

# Streamlit headers
st.markdown('<h2 id="custom-header_5">Sentiment Analysis </h2>', unsafe_allow_html=True)
st.sidebar.markdown('<h2 id="custom-header">Lujain Yousef</h2>', unsafe_allow_html=True)
st.sidebar.markdown('<h2 id="custom-header_2">Data Science Student</h2>', unsafe_allow_html=True)
st.sidebar.markdown('<h2 id="custom-header_3">Sentiment Analysis Project</h2>', unsafe_allow_html=True)



# First expander for text analysis
# Function to get sentiment analysis result
def get_sentiment_analysis_result(text):
    # Create the pipeline for DistilBERT
    distil_pipeline = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    distil_output = distil_pipeline(text)


    return distil_output

# Define image URLs
image_urls = {
    "positive": "https://i.ibb.co/dL8rDgL/Untitled-design-30.png",
    "negative": "https://i.ibb.co/RNyzgkq/Untitled-design-29.png"
}

# Streamlit app code
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        distil_output = get_sentiment_analysis_result(text) #, bert_output

        # Display the results
        st.write('DistilBERT Output: ', round(distil_output[0]['score'], 3), 'DistilBERT Label:',
                 distil_output[0]['label'])

        # Determine the label based on sentiment analysis
        sentiment_label = distil_output[0]['label']

        # Display the appropriate image based on the sentiment label
        st.image(image_urls[sentiment_label.lower()], caption=sentiment_label, width=100)


    # Text input for cleaning text
    pre = st.text_input('Clean Text: ')
    if pre:
        cleaned_text = cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                       stopwords=True, lowercase=True, numbers=True, punct=True)
        st.write('Cleaned Text: ', cleaned_text)

        distil_output= get_sentiment_analysis_result(cleaned_text)

        st.write('DistilBERT Output: ', round(distil_output[0]['score'], 3), 'DistilBERT Label:',
                 distil_output[0]['label'])

        # Determine the label based on sentiment analysis
        sentiment_label = distil_output[0]['label']

        # Display the appropriate image based on the sentiment label
        st.image(image_urls[sentiment_label.lower()], caption=sentiment_label, width=100)




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
# Streamlit app code
# Function to get VQA result
def get_vqa_result(image):
    # Visual Q&A pipeline
    vqa = pipeline(
        task='visual-question-answering',
        model='dandelin/vilt-b32-finetuned-vqa')
    result = vqa(image=image, question="Is the person happy?")
    return result

# Define image URLs
positive_image_url = "https://i.ibb.co/dL8rDgL/Untitled-design-30.png"
negative_image_url = "https://i.ibb.co/RNyzgkq/Untitled-design-29.png"

# Streamlit app code
with st.expander('Analyze Image'):
    st.write("Upload an image or provide a URL")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("Or enter image URL")

    if uploaded_file is not None:
        # Read image from uploaded file
        img = Image.open(uploaded_file)
    elif image_url:
        # Get image from URL
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
    else:
        st.warning("Please upload an image or provide an image URL.")
        st.stop()

    # Get VQA result
    result = get_vqa_result(img)

    # Calculate 'yes' and 'no' scores
    yes_score = next((res['score'] for res in result if res['answer'] == 'yes'), 0)
    no_score = next((res['score'] for res in result if res['answer'] == 'no'), 0)

    # Display the appropriate image based on the scores
    if yes_score > no_score:
        st.image(positive_image_url, caption="positive", width=100)
    else:
        st.image(negative_image_url, caption="negative", width=100)

    st.write("Result:", result)
