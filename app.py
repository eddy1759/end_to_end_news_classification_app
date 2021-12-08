import streamlit as st 
import joblib
import packages.data_processor as dp
from PIL import Image


#Load the model

news_clf = joblib.load(open("news_classification.pkl","rb"))
vectorizer = joblib.load(open("vectorizer.pickle","rb"))
#Load the category id
category_id = joblib.load(open("id_to_category.pickle","rb"))

#Main Streamlit Function
def main(title="streamlit news classification webapp".upper()):
    st.markdown(
    "<div style =background-color:red;padding:13px></div>"
    "<h1 style =color:white;text-align:center;>{}</h1>".format(title),
    unsafe_allow_html=True
    )
    img1 = Image.open("img.jpg")
    img2 = Image.open("author.jpg")
    st.image(img1, use_column_width=True)
    st.sidebar.image(img2, use_column_width=True)
    st.sidebar.subheader("Author: Edet Emmanuel Asuquo")
    st.sidebar.subheader("Project: News Classification Using Machine Learning")
    st.info('This webapp uses machine learning to classify news into distinct group'.upper())
    st.header("**ENTER TEXT**")
    texts_msg = st.text_area("Type here")
    if st.button("Classify"):
        st.text("Original text: \n{}".format(texts_msg))
        vect_text = vectorizer.transform([texts_msg]).toarray()
        prediction = news_clf.predict(vect_text)
        final_result = category_id[prediction[0]]
        st.success("  - The News is classified as: '{}'".format(final_result))

if __name__ == "__main__":
    main()



