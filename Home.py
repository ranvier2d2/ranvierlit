import streamlit as st

# Set page config once here
st.set_page_config(page_title='Ranvier - Kronika', page_icon='🧠')

# Function to load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
load_css("styles.css")

st.title("Ranvier Skills 🧠")
st.write("Welcome to the Ranvier-Kronika AI Skill sites. Use the sidebar to navigate to different pages.")
