import os
import streamlit as st
import nest_asyncio
import asyncio
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# Apply nest_asyncio to manage nested event loops
nest_asyncio.apply()

# Set page config once here
st.set_page_config(page_title='Ranvier - Kronika', page_icon='🧠')
st.title("AI Enhanced Review ✨ ")
st.write("Welcome to the Ranvier-Kronika AI Skill sites. Use the sidebar to navigate to different pages.")


# Function to load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
load_css("styles.css")

with st.expander('About this app'):
    st.markdown('''
    **What can this app do?**
    This app allows users to initiate a comprehensive research process on specific diseases using CrewAI agents. The agents will collect, analyze, and compile information into a coherent review.
    
    **How to use the app?**
    1. Enter a disease name in the input field.
    2. Click the "Start Research" button to begin the process.
    3. The results will be displayed once the research is completed.
    ''')

# Ensure there is an event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError as e:
    if "There is no current event loop in thread" in str(e):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Retrieve the API key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("GOOGLE_API_KEY environment variable not set. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

# Initialize the language model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

# Define agents with verbose mode and backstories
researcher = Agent(
    role='Researcher',
    goal='Collect comprehensive information on {disease_name}',
    tools=[],
    verbose=True,
    backstory=(
        "An experienced medical researcher with a focus on epidemiology and pathophysiology.\n"
        "To research {disease_name}, gather information on:\n"
        "1. Key clinical features - signs, symptoms, affected body systems, disease course and prognosis\n"
        "2. Epidemiology - incidence, prevalence, high risk populations, risk factors and causes\n"  
        "3. Pathophysiology - underlying biological mechanisms, impaired organ function, genetic and environmental factors\n"
        "4. Diagnostic strategies - typical diagnostic workup, key history and exam findings, lab tests and imaging studies, specialized testing"
    ),
    llm=llm,
    allow_delegation=False
)

analyst = Agent(
    role='Analyst',
    goal='Analyze and synthesize collected data on {disease_name}',
    tools=[],
    verbose=True,
    backstory=(
        "A skilled data analyst with expertise in medical data analysis and outcome prediction.\n"
        "When analyzing information on {disease_name}:\n"
        "1. Assess management approaches - treatment goals, medical and surgical therapies, multidisciplinary care\n"
        "2. Analyze complications and follow-up - major complications, monitoring and follow-up plans, factors influencing outcomes\n"
        "3. Utilize high-quality information resources - medical textbooks, journal articles, guidelines, expert opinions"
    ),
    llm=llm,
    allow_delegation=False
)

writer = Agent(
    role='Writer',
    goal='Compile findings on {disease_name} into a coherent review',
    tools=[],
    verbose=True,
    backstory=(
        "A proficient medical writer with a knack for synthesizing complex information into clear, concise documents.\n"
        "To write a comprehensive review on {disease_name}:\n"
        "1. Synthesize information to provide a complete picture of the disease\n"
        "2. Explain how {disease_name} fits into differential diagnoses for common presenting symptoms\n"
        "3. Discuss how the knowledge can be applied clinically to improve diagnostic reasoning and decision-making\n"
        "4. Use clear organization with sections on clinical features, epidemiology, pathophysiology, diagnosis, management, and complications"
    ),
    llm=llm,
    allow_delegation=False
)

# Define tasks
collect_clinical_features_task = Task(
    description='Collect information on the typical signs, symptoms, and clinical manifestations of {disease_name}',
    expected_output='A detailed list of clinical features and disease course of {disease_name}',
    agent=researcher,
    context=[]
)

determine_epidemiology_task = Task(
    description='Determine the incidence, prevalence, and risk factors of {disease_name}',
    expected_output='A summary of epidemiological data of {disease_name}',
    agent=researcher,
    context=[collect_clinical_features_task]
)

review_pathophysiology_task = Task(
    description='Review the biological mechanisms and factors leading to {disease_name}',
    expected_output='A detailed explanation of the pathophysiology of {disease_name}',
    agent=researcher,
    context=[determine_epidemiology_task]
)

familiarize_diagnostic_workup_task = Task(
    description='Familiarize with diagnostic workup, key findings, and specialized tests for {disease_name}',
    expected_output='A comprehensive list of diagnostic strategies for {disease_name}',
    agent=researcher,
    context=[review_pathophysiology_task]
)

review_management_approaches_task = Task(
    description='Review medical and surgical treatments, and multidisciplinary care for {disease_name}',
    expected_output='A summary of management approaches for {disease_name}',
    agent=analyst,
    context=[familiarize_diagnostic_workup_task]
)

recognize_complications_task = Task(
    description='Recognize complications, monitoring, and follow-up plans for {disease_name}',
    expected_output='A detailed list of complications and follow-up strategies for {disease_name}',
    agent=analyst,
    context=[review_management_approaches_task]
)

synthesize_information_task = Task(
    description='Synthesize all gathered information on {disease_name} into a comprehensive review',
    expected_output='A well-structured review document integrating knowledge into clinical reasoning for {disease_name}, including the top 5-10 clinical pearls',
    agent=writer,
    context=[
        collect_clinical_features_task,
        determine_epidemiology_task,
        review_pathophysiology_task,
        familiarize_diagnostic_workup_task,
        review_management_approaches_task,
        recognize_complications_task
    ]
)

# Create the crew
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[
        collect_clinical_features_task,
        determine_epidemiology_task,
        review_pathophysiology_task,
        familiarize_diagnostic_workup_task,
        review_management_approaches_task,
        recognize_complications_task,
        synthesize_information_task
    ],
    process=Process.sequential
)

# Streamlit input
disease_name = st.text_input("Enter disease name:", "")

if st.button("Start Research"):
    if disease_name:
        st.write(f"Researching {disease_name}...")
        inputs = {
            "disease_name": disease_name,
        }
        try:
            with st.spinner('Running CrewAI tasks...'):
                result = crew.kickoff(inputs=inputs)
                
                st.success("Research completed!")
                
                detailed_results = []
                for task in crew.tasks:
                    task_result = task.output
                    detailed_results.append({
                        "task": task.description,
                        "result": task_result
                    })
                
                # Store detailed results and research result in session state
                st.session_state['detailed_results'] = detailed_results
                st.session_state['research_result'] = result
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a disease name.")

# Show research result
if 'research_result' in st.session_state:
    st.write(st.session_state['research_result'])
    
# Show detailed results in an expander
if 'detailed_results' in st.session_state:
    with st.expander("Show detailed results"):
        for detail in st.session_state['detailed_results']:
            st.write(f"**Task:** {detail['task']}")
            st.write(f"**Result:** {detail['result']}")
            st.write("---")
