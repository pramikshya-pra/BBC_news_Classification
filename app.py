import streamlit as st
import pandas as pd 
import pickle

st.title("News Clasification")

# Load model
dbfile = open("LogisticRegression.pickle", 'rb')
model = pickle.load(dbfile)

#Taking Data from user and convert to DataFrame
news= st.text_area("Enter news for classification")

if st.button("Submit"):
    d= {'news': [news]}
    df= pd.DataFrame(d)

    # predict the news
    result = model.predict(df['news'])[0]
    st.dataframe(df)
    st.write(result)
