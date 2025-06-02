import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import pipeline


rev_df = pd.read_csv('e_com_data_cleaned.csv')  
tf = TfidfVectorizer()
x = tf.fit_transform(rev_df['text_features'])
y = rev_df['sentiment']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

page_bg_color = '''
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f5f5f5;
    color: black;
}
[data-testid="stHeader"] {
    background-color: rgba(255, 255, 255, 0.9);
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.95);
}
</style>
'''
st.markdown(page_bg_color, unsafe_allow_html=True)


st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Home', 'Dataset Viewer', 'Findings', 'Sentiment Predictor'])

if page == 'Home':
    st.title('Welcome to E-Commerce Review Analyzer')
    st.write('''This project uses Machine Learning and NLP to analyze customer reviews from an e-commerce dataset. 
                 It predicts sentiment (positive/negative), performs aspect-based analysis, and provides visual insights.''')
    st.markdown("---")
    st.markdown("Created by Mayur Patil | 📧 mayurpatil017902@gmail.com")

elif page == 'Dataset Viewer':
    st.title('Dataset Viewer')
    try:
        st.dataframe(rev_df)
    except FileNotFoundError:
        st.error("The file was not found in the directory.")

elif page == 'Findings':
    st.title('Insights')

    image_files = [f'Images/{i}.png' for i in range(1, 7)]
    select_image = st.selectbox("Select an Image to View", options=image_files)

    image_info = {
        'Images/1.png': """
**Filename**: `Images/1.png`  
This pie chart shows the sentiment distribution of customer reviews.  
Key Segments:

1-Positive Sentiment (Blue): This is the largest portion, representing 10.43K ratings, which is 49.7% of the total. This means almost half of the ratings are positive.
2-Negative Sentiment (Dark Blue): This is the second largest, with 8.31K ratings, making up 39.58% of the total. This indicates a significant portion of ratings are negative.
3-Neutral Sentiment (Orange): This is the smallest segment, with 2.25K ratings, accounting for 10.72% of the total. This suggests a smaller number of ratings are neutral.
""",
        'Images/2.png': """
**Filename**: `Images/2.png`  
This bar chart compares the number of reviews across different product categories.  
Key Segments (Review Counts by Rating):

1-Star Rating (Dark Green, Longest Bar): This is the most frequent rating, with a very high count of 13.1K reviews. This indicates a significant number of very negative experiences.
2-Star Rating (Dark Green, Above 1-Star): This rating received 1.4K reviews, indicating a moderate number of negative experiences.
3-Star Rating (Dark Green, Smallest Bar): This is the least frequent rating, with 0.9K reviews. This suggests very few customers had a neutral or average experience.
4-Star Rating (Dark Green, Below 5-Star): This rating received 1.3K reviews, showing a decent number of positive, but not perfect, experiences.
5-Star Rating (Dark Green, Second Longest Bar at Top): This is the second most frequent rating, with 4.5K reviews. This shows a notable number of very positive experiences.
""",
        'Images/3.png': """
**Filename**: `Images/3.png`  
This line graph shows review volume trends over time.  
Key Trends:

1-Initial Stability (Pre-2010): The polarity score was near zero or very low, indicating neutral or no significant sentiment activity.
2-First Major Positive Peak (Around 2012): There's a significant surge in positive polarity, peaking at nearly 400. This suggests a period of very strong positive sentiment.
3-Decline and Minor Fluctuation (2013-2015): After the peak, the polarity score drops sharply but remains positive, with a smaller bump around 2015.
4-Second Positive Wave (Around 2017-2019): Another substantial wave of positive polarity emerges, reaching a peak of around 180-190. This indicates a strong period of positive sentiment, though not as high as the 2012 peak.
5-Decline to Negative (Late 2019 - Early 2020): The polarity score starts to decline significantly, crossing the zero line and moving into negative territory. This marks a shift from generally positive to negative sentiment.
6-Sustained Negative Polarity (2020-Present): From approximately 2020 onwards, the polarity score remains consistently negative, fluctuating between -50 and -100. This suggests a period of overall negative sentiment.
""",
        'Images/4.png': """
**Filename**: `Images/4.png`  
Word cloud of most common words in positive reviews.  
Key Observations by Region:

1-Europe (Highly Concentrated): There is a very high concentration of blue circles, many of them large, particularly in Western Europe (e.g., UK, France, Germany, Spain, Italy). This suggests a significant volume of sentiment data originates from this region. The largest circle on the map appears to be in Europe, indicating the highest concentration of sentiment.
2-North America (Significant Activity): The United States and parts of Canada show several large blue circles, indicating a substantial amount of sentiment activity.
3-Asia (Moderate to Scattered): While there are many smaller circles across Asia, there are a few larger ones, notably in India and possibly parts of Southeast Asia. This suggests varying levels of sentiment data across the continent.
4-South America (Moderate Activity): Several circles are visible in South America, particularly along the coastlines, indicating a decent amount of sentiment data.
5-Australia (Visible Activity): Australia shows some visible blue circles, suggesting sentiment data from this region.
6-Africa (Scattered, Smaller): While there are some circles, they appear generally smaller and more scattered across Africa, indicating a lower volume of sentiment data compared to other major continents.
Interpreting Circle Size: Larger blue circles likely represent countries or regions with a higher volume of reviews, mentions, or overall data related to the sentiment being tracked. Smaller circles indicate less data.
""",
        'Images/5.png': """
**Filename**: `Images/5.png`  
This scatter plot shows top reviewers ranked by review count and activity frequency.  
1-Most Active Reviewers:

-The reviewer named "customer" has written the most reviews, with a count of 34. This might indicate a generic name for a large group of anonymous reviewers or a placeholder.
-"David" is the second most active reviewer, with 24 reviews.
-"Mark" follows with 20 reviews.
    
2-Moderately Active Reviewers:

-"John" and "Paul" both have 18 reviews each.
-"Chris" and "Michael" each have 16 reviews.
-"James" has 15 reviews.
-"Mike" has 13 reviews.

3-Least Active Listed Reviewer:

-The reviewer named "Consumer" has the fewest reviews among those shown, with only 6. This, like "customer," could also be a generic label.
""",
        'Images/6.png': """
**Filename**: `Images/6.png`  
This heatmap highlights correlation between different review features.  
1 - Peak Rating:

The highest average rating occurs in February, peaking at approximately 2.41.
Lowest Rating:
sThe lowest average rating occurs in August, dropping significantly to just under 2.0.

2 - Seasonal Trends:

Early Year (January-February): Ratings start relatively high and peak in February.
Spring Decline (March-May): There's a notable dip in March, followed by a slight recovery in April and May, but still lower than the February peak.
Summer Drop (June-August): Ratings generally decline through June and July, hitting their lowest point in August. This suggests a significant drop in customer satisfaction during the summer months.
Autumn Recovery (September-October): There's a clear recovery in ratings from September, peaking again in October.
Year-End Decline (November-December): Ratings slightly decline towards the end of the year, in November and December.
"""
    }

    try:
        image = Image.open(select_image)
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption=select_image)

        with col2:
            st.subheader('Insights Information')
            st.markdown(image_info.get(select_image, "No information available for this image."))

    except FileNotFoundError:
        st.error(f"Image file '{select_image}' not found.")

elif page == 'Sentiment Predictor':
    st.title('📊 Sentiment Predictor (Logistic Regression + BERT)')
    st.write("Enter a customer review below to predict the sentiment using both Logistic Regression and BERT model.")

    user_input = st.text_area("Enter Review Text", height=150)

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text before predicting.")
        else:
            user_input_vectorized = tf.transform([user_input])
            lr_prediction = model.predict(user_input_vectorized)[0]
            lr_label = "Positive" if lr_prediction == 1 else "Negative"

            bert_result = classifier(user_input)[0]
            bert_label = bert_result['label']
            bert_score = bert_result['score']

            st.markdown(f"### ✅ Logistic Regression Prediction: **{lr_label}**")
            st.markdown(f"### 🤖 BERT Prediction: **{bert_label}** (Confidence: {bert_score:.2f})")
