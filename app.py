from collections import Counter

import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import wordcloud
import pandas as pd
import seaborn as sns
from PIL import Image
from transformers import pipeline

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.markdown(
    """
    <style>
    
    sidebar .sidebar-content {
          background-color: #800080;
      }
    .stMarkdown h2{ /*to change font size of header(total messages)*/
            font-size: 21px; /* Adjust font size as needed */
    }
    .stMarkdown h1{           /*to change font size of header(num_messages, words)*/
            font-size: 28px; /* Adjust font size as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("Whatsapp chat analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")
flag = False
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    user_list = df['user'].unique().tolist()
    if "group notification" in user_list:
        user_list.remove("group_notification")
    user_list.sort()
    user_list.insert(0,  'overall')
    selected_user = st.sidebar.selectbox("Whatsapp analysis wrt", user_list)
    st.dataframe(df)

    if st.sidebar.button("show analysis"):
        st.title("TOP ANALYSIS")
        num_messages, words, num_media, total_emojis, total_links = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header(" :white[--> Total Messages]")
            st.title(num_messages)
        with col2:
            st.header("--> Total Words")
            st.title(words)
        with col3:
            st.header("--> Total Media")
            st.title(num_media)
        with col4:
            st.header("--> Total Emojis")
            st.title(total_emojis)
        with col2:
            st.header("--> Total links")
            st.title(total_links)

        # finding busy users in group level
        if selected_user == 'overall':
            st.title('Most busy users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            custom_colors = ['green', 'blue', 'orange', 'red', 'yellow']

            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color=custom_colors)
                plt.gcf().set_facecolor('lightblue')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        st.title("Word Cloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)
# most common users
        st.title("Most Common Words")
        most_common_df = helper.most_common_users(selected_user, df)
        fig, ax = plt.subplots()
        custom_colors = ['green', 'blue', 'orange', 'red', 'yellow', 'Crimson', 'Lime', 'Magenta', 'Teal', 'Brown']
        ax.bar(most_common_df[0], most_common_df[1], color=custom_colors)
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

# emoji analysis
        all_emojis = helper.emoji_helper(selected_user, df)
        emoji_df = pd.DataFrame(Counter(all_emojis).most_common(len(Counter(all_emojis))))
        emoji_counts = Counter(all_emojis)
        st.title("Emoji Analysis")

        labels = emoji_counts.keys()
        sizes = emoji_counts.values()

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            st.pyplot(fig)

# monthly timeline analysis

        monthly_timeline = helper.monthly_timeline(selected_user, df)
        st.title("Monthly Time Line")
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(monthly_timeline['time'], monthly_timeline['message'])
        plt.gcf().set_facecolor('lightblue')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

# daily timeline analysis

        daily_timeline = helper.daily_timeline(selected_user, df)
        st.title("Daily Time Line")
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.plot(daily_timeline['only_date'], daily_timeline['message'])
        plt.gcf().set_facecolor('lightblue')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

# activity map
        st.title("Activity map")
        col1, col2 = st.columns(2)
        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Activity heatmap")
        activity_heatmap = helper.heatmap(selected_user, df)
        plt.figure(figsize=(20, 6))
        fig, ax = plt.subplots()
        ax = sns.heatmap(activity_heatmap.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0))
        st.pyplot(fig)


        st.title("Spam prediction")
        res = helper.spam_verification(selected_user, df)
        if res == 'notspam':
            st.header("The text messages are not spam")

            # Load the image
            image = Image.open("C:\\Users\\Admin\\Desktop\\whatsapp chat analyzer\\whatsapp project\\not-spam.jpg")
            # Display the image
            st.image(image, use_column_width=True)
        else:
            st.header("The text messages are spam")
            # Load the image
            image = Image.open("C:\\Users\\Admin\\Desktop\\whatsapp chat analyzer\\whatsapp project\\spam.jpg")
            # Display the image
            st.image(image, use_column_width=True)

        st.title("Predicted messages for selected month")
    if 'selected_month' not in st.session_state:
        st.session_state.selected_month = None
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False
    month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                      'November', 'December']
    selected_month = st.selectbox("select the month ", month_list)
    timeline, month, message = helper.message_prediction(selected_user, df, selected_month)
    if st.button("predict"):
        st.header("prediction for the month ")
        st.header(selected_month)
        st.session_state.selected_option = selected_month
        st.session_state.button_clicked = True

        # Display the selected option if the button was clicked
    if st.session_state.button_clicked:
        st.write(int(message))

        st.title("Sentiment Analysis of Text")
        pos_percent, neg_percent, neu_percent = helper.sentiment(selected_user, df)
        if pos_percent > neg_percent:
            if pos_percent > neu_percent:
                st.header("The sentiment of most of the text is positive")
            else:
                st.header("The sentiment of most of the text is neutral")
        else:
            if neg_percent > neu_percent:
                st.header("The sentiment of most of the text is negative")
            else:
                st.header("The sentiment of most of the text is neutral")