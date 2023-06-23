import pickle
import streamlit as st
model = pickle.load(open("fake.pkl","rb"))
model = pickle.load(open("true.pkl","rb"))
cv = pickle.load(open("vectorizer.pkl","rb"))

def main():
    st.title("EMAIL SPAM FILTER MODEL")
    st.subheader("Made by Gurleen kaur")
    msg = st.text_input("Enter the text: ")
    if st.button("Predict"):
        data = [msg]
        vect = cv.transform(data).toarray()
        prediction = model.predict(vect)
        result = prediction[0]

        if result == 1:
            st.error("This is fake")
        else:
            st.success("This is not fake")




main()
        
    
