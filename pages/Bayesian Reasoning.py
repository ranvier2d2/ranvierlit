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
st.set_page_config(page_title='Clinical Diagnostic Assistant - Page 3', page_icon='🩺')

st.title('🩺 Clinical Diagnostic Assistant')

# Load the CSS file (ensure this path is correct relative to main.py)
css_file_path = Path(__file__).parent.parent / "styles.css"
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css(css_file_path)

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
    goal='Collect comprehensive information on the patient\'s clinical history and chief complaint',
    tools=[],
    verbose=True,
    backstory=(
        "An experienced medical researcher with a focus on epidemiology and pathophysiology.\n"
        "To research the patient's clinical history and chief complaint, gather information on:\n"
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
    goal='Analyze and synthesize collected data on the patient\'s clinical history and chief complaint',
    tools=[],
    verbose=True,
    backstory=(
        "A skilled data analyst with expertise in medical data analysis and outcome prediction.\n"
        "When analyzing information on the patient's clinical history and chief complaint:\n"
        "1. Assess management approaches - treatment goals, medical and surgical therapies, multidisciplinary care\n"
        "2. Analyze complications and follow-up - major complications, monitoring and follow-up plans, factors influencing outcomes\n"
        "3. Utilize high-quality information resources - medical textbooks, journal articles, guidelines, expert opinions"
    ),
    llm=llm,
    allow_delegation=False
)

writer = Agent(
    role='Writer',
    goal='Compile findings on the patient\'s clinical history and chief complaint into a coherent diagnostic framework',
    tools=[],
    verbose=True,
    backstory=(
        "A proficient medical writer with a knack for synthesizing complex information into clear, concise documents.\n"
        "To write a comprehensive diagnostic framework on the patient's clinical history and chief complaint:\n"
        "1. Synthesize information to provide a complete picture of the patient's condition\n"
        "2. Explain how the patient's symptoms fit into differential diagnoses for common presenting symptoms\n"
        "3. Discuss how the knowledge can be applied clinically to improve diagnostic reasoning and decision-making\n"
        "4. Use clear organization with sections on clinical features, epidemiology, pathophysiology, diagnosis, management, and complications"
    ),
    llm=llm,
    allow_delegation=False
)

# Define tasks with context and error handling
initial_assessment_task = Task(
    description='Collect information on the patient\'s chief complaint and history of present illness (HPI)',
    expected_output='A detailed account of the patient\'s chief complaint and HPI',
    agent=researcher,
    context=[]
)

comprehensive_medical_history_task = Task(
    description='Gather comprehensive medical history, including past medical history, medications, allergies, family history, and social history',
    expected_output='A comprehensive medical history of the patient',
    agent=researcher,
    context=[initial_assessment_task]
)

review_of_systems_task = Task(
    description='Conduct a review of systems (ROS) to identify any additional symptoms across different body systems',
    expected_output='A detailed review of systems (ROS) for the patient',
    agent=researcher,
    context=[comprehensive_medical_history_task]
)

physical_examination_task = Task(
    description='Perform a targeted physical examination based on the information gathered so far',
    expected_output='A detailed report of the physical examination findings',
    agent=researcher,
    context=[review_of_systems_task]
)

differential_diagnosis_task = Task(
    description='Generate a list of possible diagnoses (differential diagnosis) based on the patient\'s symptoms, history, and physical examination findings',
    expected_output='A prioritized list of possible diagnoses',
    agent=analyst,
    context=[physical_examination_task]
)

bayesian_reasoning_task = Task(
    description='Apply Bayesian reasoning to refine the differential diagnosis using known and unknown baseline probabilities',
    expected_output='A refined differential diagnosis using Bayesian reasoning',
    agent=analyst,
    context=[differential_diagnosis_task]
)

diagnostic_testing_task = Task(
    description='Order and interpret diagnostic tests to gather more objective data',
    expected_output='A detailed report of diagnostic test results and their interpretation',
    agent=analyst,
    context=[bayesian_reasoning_task]
)

synthesize_diagnostic_framework_task = Task(
    description='Synthesize all gathered information into a comprehensive diagnostic framework for the patient\'s clinical history and chief complaint',
    expected_output='A well-structured diagnostic framework integrating all gathered information, including the top 5-10 clinical pearls',
    agent=writer,
    context=[
        initial_assessment_task,
        comprehensive_medical_history_task,
        review_of_systems_task,
        physical_examination_task,
        differential_diagnosis_task,
        bayesian_reasoning_task,
        diagnostic_testing_task
    ]
)

# Create the crew
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[
        initial_assessment_task,
        comprehensive_medical_history_task,
        review_of_systems_task,
        physical_examination_task,
        differential_diagnosis_task,
        bayesian_reasoning_task,
        diagnostic_testing_task,
        synthesize_diagnostic_framework_task
    ],
    process=Process.sequential
)

# Streamlit inputs
clinical_history = st.text_area("Enter clinical history:", "")
chief_complaint = st.text_input("Enter chief complaint:", "")

if st.button("Start Clinical Assessment"):
    if clinical_history and chief_complaint:
        st.write(f"Assessing clinical history and chief complaint...")
        inputs = {
            "clinical_history": clinical_history,
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
        st.warning("Please enter both clinical history and chief complaint.")

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
