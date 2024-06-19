import os
import streamlit as st
import nest_asyncio
import asyncio
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# Apply nest_asyncio to manage nested event loops
nest_asyncio.apply()

# Set page config
st.set_page_config(page_title='Medical Assistant - Page 2', page_icon='🩺')

st.title('🩺 Chief Complaint Medical Assistant')

# Load the CSS file (ensure this path is correct relative to main.py)
css_file_path = Path(__file__).parent.parent / "styles.css"
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css(css_file_path)

# Retrieve the API key from Streamlit secrets
google_api_key = st.secrets["GOOGLE_API_KEY"]

if not google_api_key:
    st.error("GOOGLE_API_KEY environment variable not set. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

# Initialize the language model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

# Define agents with verbose mode and backstories
history_taker = Agent(
    role='History Taker',
    goal='Gather comprehensive patient history based on {chief_complaint}',
    tools=[],
    verbose=True,
    backstory=(
        "An experienced clinician skilled in obtaining detailed patient history.\n"
        "To gather history for {chief_complaint}, focus on:\n"
        "1. Onset, duration, progression, quality, severity, location, aggravating/relieving factors, and associated symptoms\n"
        "2. Past medical history, medications, allergies, family history, and social history\n"
        "3. Review of systems to identify potential symptoms across different body systems"
    ),
    llm=llm,
    allow_delegation=False
)

examiner = Agent(
    role='Physical Examiner',
    goal='Perform targeted physical examination based on {chief_complaint} and patient history',
    tools=[],
    verbose=True,
    backstory=(
        "A skilled clinician experienced in conducting physical examinations.\n"
        "When examining a patient with {chief_complaint}:\n"
        "1. Focus on relevant body systems based on the history and chief complaint\n"
        "2. Look for signs that can help narrow down the differential diagnosis\n"
        "3. Document findings thoroughly for further analysis"
    ),
    llm=llm,
    allow_delegation=False
)

diagnostician = Agent(
    role='Diagnostician',
    goal='Generate differential diagnosis and recommend diagnostic tests for {chief_complaint}',
    tools=[],
    verbose=True,
    backstory=(
        "An expert diagnostician well-versed in generating differential diagnoses and selecting appropriate tests.\n"
        "When approaching a patient with {chief_complaint}:\n"
        "1. Generate a list of possible diagnoses based on history and examination findings\n"
        "2. Prioritize diagnoses based on likelihood, severity, and urgency\n"
        "3. Apply Bayesian reasoning to refine probabilities and guide diagnostic testing"
    ),
    llm=llm,
    allow_delegation=False
)

# Define tasks with context and error handling
gather_history_task = Task(
    description='Gather comprehensive patient history based on {chief_complaint}',
    expected_output='A detailed patient history, including HPI, PMH, medications, allergies, family history, social history, and ROS',
    agent=history_taker,
    context=[]
)

perform_examination_task = Task(
    description='Perform targeted physical examination based on {chief_complaint} and patient history',
    expected_output='A thorough physical examination report focusing on relevant body systems',
    agent=examiner,
    context=[gather_history_task]
)

generate_differential_diagnosis_task = Task(
    description='Generate differential diagnosis and recommend diagnostic tests for {chief_complaint}',
    expected_output='A prioritized list of potential diagnoses and recommended diagnostic tests, with reasoning based on Bayesian principles',
    agent=diagnostician,
    context=[gather_history_task, perform_examination_task]
)

# Create the crew
crew = Crew(
    agents=[history_taker, examiner, diagnostician],
    tasks=[
        gather_history_task,
        perform_examination_task,
        generate_differential_diagnosis_task
    ],
    process=Process.sequential
)

# Streamlit input
chief_complaint = st.text_input("Enter chief complaint:", "")

if st.button("Start Medical Assessment"):
    if chief_complaint:
        st.write(f"Assessing {chief_complaint}...")
        inputs = {
            "chief_complaint": chief_complaint,
        }
        try:
            with st.spinner('Running CrewAI tasks...'):
                result = crew.kickoff(inputs=inputs)
                
                st.success("Assessment completed!")
                
                detailed_results = []
                for task in crew.tasks:
                    task_result = task.output
                    detailed_results.append({
                        "task": task.description,
                        "result": task_result
                    })
                
                # Store detailed results and assessment result in session state
                st.session_state['detailed_results'] = detailed_results
                st.session_state['assessment_result'] = result
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a chief complaint.")

# Show assessment result
if 'assessment_result' in st.session_state:
    st.write(st.session_state['assessment_result'])
    
# Show detailed results in an expander
if 'detailed_results' in st.session_state:
    with st.expander("Show detailed results"):
        for detail in st.session_state['detailed_results']:
            st.write(f"**Task:** {detail['task']}")
            st.write(f"**Result:** {detail['result']}")
            st.write("---")
