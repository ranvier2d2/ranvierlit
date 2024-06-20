import os
import logging
import streamlit as st
import nest_asyncio
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    handlers=[logging.StreamHandler()])

# Apply nest_asyncio to manage nested event loops
nest_asyncio.apply()

# Set page config
st.set_page_config(page_title='Clinical Diagnostic Assistant - Page 3',
                   page_icon='🩺')
st.title('🩺 Clinical Diagnostic Assistant')

# Load the CSS file (ensure this path is correct relative to main.py)
css_file_path = Path(__file__).parent.parent / "styles.css"


def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logging.info(f"Loaded CSS file: {file_name}")
    except FileNotFoundError:
        logging.error(f"CSS file {file_name} not found.")
        st.error(f"CSS file {file_name} not found.")


load_css(css_file_path)

# Retrieve the API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error(
        "OPENAI_API_KEY environment variable not set. Please set the OPENAI_API_KEY environment variable."
    )
    logging.error("OPENAI_API_KEY environment variable not set.")
    st.stop()

logging.info("OPENAI_API_KEY environment variable retrieved successfully.")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
# Define agents with verbose mode and backstories
researcher = Agent(
    role='Researcher',
    goal=
    'Collect comprehensive information on the patient\'s clinical history and chief complaint: {clinical_history}, {chief_complaint}',
    tools=[],
    verbose=True,
    backstory=
    ("An experienced medical researcher with a focus on epidemiology and pathophysiology.\n"
     "To research the patient's clinical history and chief complaint, gather information on:\n"
     "1. Key clinical features - signs, symptoms, affected body systems, disease course and prognosis\n"
     "2. Epidemiology - incidence, prevalence, high risk populations, risk factors and causes\n"
     "3. Pathophysiology - underlying biological mechanisms, impaired organ function, genetic and environmental factors\n"
     "4. Diagnostic strategies - typical diagnostic workup, key history and exam findings, lab tests and imaging studies, specialized testing"
     ),
    llm=llm,  # Specify gpt-4o model here
    allow_delegation=False)

analyst = Agent(
    role='Analyst',
    goal=
    'Analyze and synthesize collected data on the patient\'s clinical history and chief complaint: {clinical_history}, {chief_complaint}',
    tools=[],
    verbose=True,
    backstory=
    ("A skilled data analyst with expertise in medical data analysis and outcome prediction.\n"
     "When analyzing information on the patient's clinical history and chief complaint:\n"
     "1. Assess management approaches - treatment goals, medical and surgical therapies, multidisciplinary care\n"
     "2. Analyze complications and follow-up - major complications, monitoring and follow-up plans, factors influencing outcomes\n"
     "3. Utilize high-quality information resources - medical textbooks, journal articles, guidelines, expert opinions"
     ),
    llm=llm,
    allow_delegation=False)

writer = Agent(
    role='Writer',
    goal=
    'Compile findings on the patient\'s clinical history and chief complaint into a coherent diagnostic framework: {clinical_history}, {chief_complaint}',
    tools=[],
    verbose=True,
    backstory=
    ("A proficient medical writer with a knack for synthesizing complex information into clear, concise documents.\n"
     "To write a comprehensive diagnostic framework on the patient's clinical history and chief complaint:\n"
     "1. Synthesize information to provide a complete picture of the patient's condition\n"
     "2. Explain how the patient's symptoms fit into differential diagnoses for common presenting symptoms\n"
     "3. Discuss how the knowledge can be applied clinically to improve diagnostic reasoning and decision-making\n"
     "4. Use clear organization with sections on clinical features, epidemiology, pathophysiology, diagnosis, management, and complications"
     ),
    llm=llm,  # Specify gpt-4o model here
    allow_delegation=False)

logging.info("Agents initialized successfully.")

# Define tasks with context and error handling
initial_assessment_task = Task(
    description=
    'Collect information on the patient\'s chief complaint and history of present illness (HPI): {clinical_history}, {chief_complaint}',
    expected_output=
    'A detailed account of the patient\'s chief complaint and HPI: {clinical_history}, {chief_complaint}',
    agent=researcher,
    context=[])

comprehensive_medical_history_task = Task(
    description=
    'Gather comprehensive medical history, including past medical history, medications, allergies, family history, and social history for {clinical_history}, {chief_complaint}',
    expected_output=
    'A comprehensive medical history of the patient: {clinical_history}, {chief_complaint}',
    agent=researcher,
    context=[initial_assessment_task])

review_of_systems_task = Task(
    description=
    'Conduct a review of systems (ROS) to identify any additional symptoms across different body systems: {clinical_history}, {chief_complaint}',
    expected_output=
    'A detailed review of systems (ROS) for the patient: {clinical_history}, {chief_complaint}',
    agent=researcher,
    context=[comprehensive_medical_history_task])

physical_examination_task = Task(
    description=
    'Perform a targeted physical examination based on the information gathered so far: {clinical_history}, {chief_complaint}',
    expected_output=
    'A detailed report of the physical examination findings: {clinical_history}, {chief_complaint}',
    agent=researcher,
    context=[review_of_systems_task])

differential_diagnosis_task = Task(
    description=
    'Generate a list of possible diagnoses (differential diagnosis) based on the patient\'s symptoms, history, and physical examination findings: {clinical_history}, {chief_complaint}',
    expected_output=
    'A prioritized list of possible diagnoses: {clinical_history}, {chief_complaint}',
    agent=analyst,
    context=[physical_examination_task])

bayesian_reasoning_task = Task(
    description=
    'Apply Bayesian reasoning to refine the differential diagnosis using known and unknown baseline probabilities: {clinical_history}, {chief_complaint}',
    expected_output=
    'A refined differential diagnosis using Bayesian reasoning: {clinical_history}, {chief_complaint}',
    agent=analyst,
    context=[differential_diagnosis_task])

diagnostic_testing_task = Task(
    description=
    'Order and interpret diagnostic tests to gather more objective data: {clinical_history}, {chief_complaint}',
    expected_output=
    'A detailed report of diagnostic test results and their interpretation: {clinical_history}, {chief_complaint}',
    agent=analyst,
    context=[bayesian_reasoning_task])

synthesize_diagnostic_framework_task = Task(
    description=
    'Synthesize all gathered information into a comprehensive diagnostic framework for the patient\'s clinical history and chief complaint: {clinical_history}, {chief_complaint}',
    expected_output=
    'A well-structured diagnostic framework integrating all gathered information, including the top 5-10 clinical pearls: {clinical_history}, {chief_complaint}',
    agent=writer,
    context=[
        initial_assessment_task, comprehensive_medical_history_task,
        review_of_systems_task, physical_examination_task,
        differential_diagnosis_task, bayesian_reasoning_task,
        diagnostic_testing_task
    ])

logging.info("Tasks defined successfully.")

# Create the crew
crew = Crew(agents=[researcher, analyst, writer],
            tasks=[
                initial_assessment_task, comprehensive_medical_history_task,
                review_of_systems_task, physical_examination_task,
                differential_diagnosis_task, bayesian_reasoning_task,
                diagnostic_testing_task, synthesize_diagnostic_framework_task
            ],
            process=Process.sequential)

logging.info("Crew created successfully.")

# Streamlit inputs
clinical_history = st.text_area("Enter clinical history:", "")
chief_complaint = st.text_input("Enter chief complaint:", "")

if st.button("Start Clinical Assessment"):
    if clinical_history and chief_complaint:
        st.write(f"Assessing clinical history and chief complaint...")
        logging.info("Starting clinical assessment with provided inputs.")
        inputs = {
            "clinical_history": clinical_history,
            "chief_complaint": chief_complaint,
        }
        try:
            with st.spinner('Running CrewAI tasks...'):
                logging.info("Running CrewAI tasks...")
                result = crew.kickoff(inputs=inputs)
                st.success("Assessment completed!")
                logging.info("Assessment completed successfully.")

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
                logging.info("Results stored in session state.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"An error occurred during assessment: {str(e)}")
    else:
        st.warning("Please enter both clinical history and chief complaint.")
        logging.warning("Clinical history or chief complaint not provided.")

# Show assessment result
if 'assessment_result' in st.session_state:
    st.write(st.session_state['assessment_result'])
    logging.info("Displayed assessment result.")

# Show detailed results in an expander
if 'detailed_results' in st.session_state:
    with st.expander("Show detailed results"):
        for detail in st.session_state['detailed_results']:
            st.write(f"**Task:** {detail['task']}")
            st.write(f"**Result:** {detail['result']}")
            st.write("---")
        logging.info("Displayed detailed results.")
