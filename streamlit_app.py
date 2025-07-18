import streamlit as st

st.title("Simple Streamlit Form")

with st.form(key="my_form"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    submit_button = st.form_submit_button("Submit")

    if submit_button:
        st.success(f"Hello {name}! Your email ({email}) was submitted!")