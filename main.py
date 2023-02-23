import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import altair as alt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

st.set_option('deprecation.showPyplotGlobalUse', False)

data = pd.read_csv("Instagram.csv", encoding = 'latin1')
data = data.dropna()


home_chart = ff.create_distplot([data["From Home"]], ["Distribution of Impressions From Home"], show_hist=False)
hashtag_chart = ff.create_distplot([data["From Hashtags"]], ["Distribution of Impressions From Hashtags"], show_hist=False)
explore_chart = ff.create_distplot([data["From Explore"]], ["Distribution of Impressions From Explore"], show_hist=False)
total_chart = ff.create_distplot([data["From Home"], data["From Hashtags"], data["From Explore"]], ["Distribution of Impressions From Home", "Distribution of Impressions from Hashtags", "Distribution of Impressions from Explore"], show_hist=False)


home_data = data["From Home"].sum()
hashtag_data = data["From Hashtags"].sum()
explore_data = data["From Explore"].sum()
other_data = data["From Other"].sum()
labels = ["From Home", "From Hashtags", "From Explore", "Other"]
values = [home_data, hashtag_data, explore_data, other_data]
total_pie = px.pie(data, values=values, names=labels, hole=0.5)


st.title("Instagram Reach Analysis")
st.caption("Dataset provided by Aman Kharwal @amankharwal.official")

st.dataframe(data)

st.header("Distribution of Impressions, and where they came from:")

home, hashtags, explore, total = st.tabs(["Home", "Hashtags", "Explore", "Total"])

home.write("Looking at the impressions from home, we can see that it's hard for this profile to reach all of their followers daily.")
home.plotly_chart(home_chart, use_container_width=True)

hashtags.write("Looking at hashtag impressions shows that not all posts can be reached using hashtags, but many new users can be reached from them")
hashtags.plotly_chart(hashtag_chart, use_container_width=True)

explore.write("By looking at the impressions this profile received from the explore section, we can observe that Instagram does not recommend their posts much to the users. Some posts have received a good reach from the explore section, but it’s still very low compared to the reach they receive from hashtags.")
explore.plotly_chart(explore_chart, use_container_width=True)

total.plotly_chart(total_chart, use_container_width=True)
total.plotly_chart(total_pie, use_container_width=True)


st.header("Analyzing Content")
st.subheader("Now let’s analyze the content of their Instagram posts. The dataset should two columns, namely caption and hashtags, which will help us understand the kind of content this profile posts on Instagram.")
st.write("Here is a wordcloud for both the Caption and Hashtags columns, to see which words are used the most")

col1, col2 = st.columns(2)

with col1:
    text = " ".join(i for i in data.Caption)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    col1.pyplot()
    st.write("Caption")

with col2:
    text = " ".join(i for i in data.Hashtags)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    col2.pyplot()
    st.write("Hashtags")


st.header("Analyzing Relationships")
relashionship = st.selectbox("Choose the relationship",
                             ("Likes X Impressions",
                             "Comments X Impressions",
                             "Shares X Impressions",
                             "Saves X Impressions"))

match relashionship:
    case 'Likes X Impressions':
        figure = px.scatter(data, x="Impressions", y="Likes", trendline="ols")
        st.plotly_chart(figure, use_container_width=True)
        st.write("There is a linear relationship between the number of likes and the reach they got on Instagram.")
    case 'Comments X Impressions':
        figure = px.scatter(data, x="Impressions", y="Comments", trendline="ols")
        st.plotly_chart(figure, use_container_width=True)
        st.write("It looks like the number of comments we get on a post doesn’t affect its reach.")
    case 'Shares X Impressions':
        figure = px.scatter(data, x="Impressions", y="Shares", trendline="ols")
        st.plotly_chart(figure, use_container_width=True)
        st.write("A more number of shares will result in a higher reach, but shares don’t affect the reach of a post as much as likes do.")
    case 'Saves X Impressions':
        figure = px.scatter(data, x="Impressions", y="Saves", trendline="ols")
        st.plotly_chart(figure, use_container_width=True)
        st.write("There is a linear relationship between the number of times their post is saved and the reach of the post.")

st.header("Analyzing Conversion Rate")
st.markdown("In Instagram, conversation rate means how many followers you are getting from the number of profile visits from a post. The formula that you can use to calculate conversion rate is **(Follows/Profile Visits) * 100.**")

conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100

code = '''conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100'''

st.code(code, language='python')

st.write(f'Result: {conversion_rate}')

st.write("So the conversation rate of their account is 41% which sounds like a very good conversation rate.")

st.subheader("Now we will use a machine learning model to predict the reach of a particular instagram post based on the following parameters")

x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)

with st.form("Model Parameters"):
    likes = st.number_input("Number of Likes", value=282.0)
    saves = st.number_input("Number of Saves", value= 233.0)
    comments = st.number_input("Number of Comments", value= 4.0)
    shares = st.number_input("Number of Shares", value= 9.0)
    profile_visits = st.number_input("Number of Profile Visits", value= 165.0)
    follows = st.number_input("Number of Follows", value= 54.0)

    submitted = st.form_submit_button("Submit")
    if submitted:
        features = np.array([[likes, saves, comments, shares, profile_visits, follows]])
        result = model.predict(features)
        st.write(f"Calculated Reach: {result}")


