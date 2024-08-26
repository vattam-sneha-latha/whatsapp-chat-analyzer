import re
from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import emoji
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from transformers import pipeline
def fetch_stats(selected_user, df):
    if selected_user != "overall":
        df = df[df['user'] == selected_user]
    # 1. fetch total no.of messages
    num_messages = df.shape[0]
    # 2. fetch total no.of words
    words = []
    for messages in df['message']:
        words.extend(messages.split())

    # 3. fetch no.of media shared
    num_media = df[df['message'] == '<Media omitted>\n'].shape[0]

    # 4.count no.of emojis
    emoji_pattern = re.compile("["
                               "\U0001F600-\U0001F64F"  # emoticons
                               "\U0001F300-\U0001F5FF"  # symbols & pictographs
                               "\U0001F680-\U0001F6FF"  # transport & map symbols
                               "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "\U00002500-\U00002BEF"  # chinese char
                               "\U00002702-\U000027B0"
                               "\U00002702-\U000027B0"
                               "\U000024C2-\U0001F251"
                               "\U0001f926-\U0001f937"
                               "\U00010000-\U0010ffff"
                               "\u200d"
                               "\u2640-\u2642"
                               "\u2600-\u2B55"
                               "\u23cf"
                               "\u23e9"
                               "\u231a"
                               "\u3030"
                               "\ufe0f"
                               "]+", flags=re.UNICODE)

    total_emojis = 0
    for message in df['message']:
        emojis_in_message = re.findall(emoji_pattern, message)
        total_emojis += len(emojis_in_message)

    # 5. find no.of links
    extractor = URLExtract()
    links = []
    for messages in df['message']:
        links.extend(extractor.find_urls(messages))
    return num_messages, len(words), num_media, total_emojis, len(links)


def most_busy_users(df):
    values_to_remove = ['group_notification']
    mask = df['user'].isin(values_to_remove)
    df = df[~mask]
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'user': 'name', 'count': 'percentage'})
    styled_df = df.style.set_properties(**{'background-color': 'orange', 'color': 'blue'})
    return x, styled_df


# word cloud
def create_wordcloud(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    wc = WordCloud(width=400, height=400, min_font_size=10, background_color='pink')
    combined_text = ' '.join(df['message'])
    f = open('C:/Users/Admin/Desktop/whatsapp chat analyzer/whatsapp project/stop_hinglish.txt', 'r')
    stop_words = f.read()
    words = combined_text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    df_wc = wc.generate(filtered_text)
    return df_wc


# most common words
def most_common_users(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp = temp[temp['message'] != '<media omitted>\n']
    f = open('C:/Users/Admin/Desktop/whatsapp chat analyzer/whatsapp project/stop_hinglish.txt', 'r')
    stop_words = f.read()
    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    most_common_df = pd.DataFrame(Counter(words).most_common(10))
    return most_common_df


# emoji analysis


def find_emojis(text):
    emojis_list = []
    emoji_info_list = emoji.emoji_list(text)
    for item in emoji_info_list:
        emojis_list.append(item['emoji'])
    return emojis_list


def emoji_helper(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    all_emojis = []
    for message in df['message']:
        all_emojis.extend(find_emojis(message))
    return all_emojis


def monthly_timeline(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    df['month_num'] = df['date'].dt.month
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    df['only_date'] = df['date'].dt.date
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline


def week_activity_map(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    df['day_name'] = df['date'].dt.day_name()
    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()


# heat map
def heatmap(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    period = []
    df['day_name'] = df['date'].dt.day_name()
    for hour in df['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))
    df['period'] = period  # creating new column period
    return df

url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Encode the labels (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_tfidf)

def predict_spam(message):
    message_tfidf = vectorizer.transform([message])
    prediction = clf.predict(message_tfidf)
    return 'spam' if prediction[0] == 1 else 'notspam'


def spam_verification(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    text = ""
    for messages in df['message']:
        text = (text + messages)
    return predict_spam(text)

def message_prediction(selected_user, df, selected_month):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    df['month_num'] = df['date'].dt.month
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    # Encode month names to numerical values
    label_encoder = LabelEncoder()
    timeline['month_encoded'] = label_encoder.fit_transform(timeline['month'])
    # Features and target variable
    X = timeline[['month_num']]  # Feature: month
    y = timeline['message']  # Target: number of text messages
    print(X.shape)
    print(y.shape)
    # Split the data into training and testing sets (optional, for a simple example we use all data for training)
    # Split the data into training and testing sets (optional)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    month_dict = {}
    month_dict['January'] = 1
    month_dict['February'] = 2
    month_dict['March'] = 3
    month_dict['April'] = 4
    month_dict['May'] = 5
    month_dict['June'] = 6
    month_dict['July'] = 7
    month_dict['August'] = 8
    month_dict['September'] = 9
    month_dict['October'] = 10
    month_dict['November'] = 11
    month_dict['December'] = 12
    month = month_dict[selected_month]
    return timeline, month+1, predict_text_messages((month+1)%12, model)


def predict_text_messages(month, model):
    print(month)
    predicted_messages = model.predict([[month]])[0]

    return predicted_messages

def sentiment(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    sentiment_analysis = pipeline("sentiment-analysis")
    text = []
    for messages in df['message']:
        text.append(messages)
    positive_count = 0
    negative_count = 0
    for message in text:
        result = sentiment_analysis(message)[0]
        if result['label'] == 'POSITIVE':
            positive_count += 1
        elif result['label'] == 'NEGATIVE':
            negative_count += 1
    total_messages = len(text)
    positive_percentage = (positive_count / total_messages) * 100
    negative_percentage = (negative_count / total_messages) * 100
    neutral_percentage = 100 - (positive_percentage + negative_percentage)
    return positive_percentage, negative_percentage, neutral_percentage

