import streamlit as st
from pathlib import Path

# Set page config once here
st.set_page_config(page_title='Ranvier - Kronika', page_icon='🧠')


# Function to load CSS
def load_css(file_name):
    css_path = Path(file_name)
    if css_path.is_file():
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.error(
            f"CSS file {file_name} not found. Please ensure it exists in the correct path."
        )


# Load the CSS file
load_css("styles.css")

st.title("Ranvier Skills 🧠")
st.write(
    "Welcome to the Ranvier-Kronika AI Skill sites. Use the sidebar to navigate to different pages."
)

# This part is essential for Replit to recognize the script
if __name__ == "__main__":
    # Main function to run the Streamlit app
    st.write("Hello, Streamlit!")
