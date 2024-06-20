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

# Define agents with enhanced roles, backstories, and goals
head_internal_medicine = Agent(
    role='Head of Internal Medicine',
    goal=
    'Collect and document detailed information on the patient\'s clinical history and chief complaint: {clinical_history}, {chief_complaint}. '
    'Identify immediate findings and suggest areas for further investigation.',
    tools=[],
    verbose=True,
    backstory=
    ("An experienced internal medicine specialist with a comprehensive understanding of various diseases and conditions. "
     "You lead the internal medicine department, ensuring accurate and thorough initial patient assessments."
     ),
    llm=llm,  # Specify gpt-4o model here
    allow_delegation=False)

semiology_expert = Agent(
    role='Semiology Expert',
    goal=
    'Gather a thorough medical history and conduct a detailed review of systems for {clinical_history}, {chief_complaint}. '
    'Identify gaps or areas needing further investigation with rationales.',
    tools=[],
    verbose=True,
    backstory=
    ("A specialist in semiology with extensive experience in understanding and interpreting symptoms and signs of diseases. "
     "You focus on comprehensive medical history taking and symptom review."),
    llm=llm,
    allow_delegation=False)

physical_exam_specialist = Agent(
    role='Physical Examination Specialist',
    goal=
    'Perform targeted physical examinations based on gathered information for {clinical_history}, {chief_complaint}. '
    'Suggest additional areas to examine, providing explanations based on the clinical context.',
    tools=[],
    verbose=True,
    backstory=
    ("An expert in conducting physical examinations with a keen eye for detail. "
     "Your expertise lies in identifying physical signs and correlating them with clinical symptoms."
     ),
    llm=llm,
    allow_delegation=False)

chief_differential_diagnosis = Agent(
    role='Chief of Differential Diagnosis Department',
    goal=
    'Generate and refine differential diagnoses using collected data for {clinical_history}, {chief_complaint}. '
    'Apply Bayesian reasoning and interpret diagnostic test results to provide comprehensive diagnostic insights.',
    tools=[],
    verbose=True,
    backstory=
    ("A seasoned medical analyst and head of the differential diagnosis department. "
     "You specialize in analyzing clinical data and refining diagnoses using Bayesian reasoning."
     ),
    llm=llm,
    allow_delegation=False)

clinical_doc_specialist = Agent(
    role='Clinical Documentation Specialist',
    goal=
    'Compile and structure all gathered information into a comprehensive diagnostic framework for {clinical_history}, {chief_complaint}. '
    'Highlight key clinical points and provide rationales for diagnostic conclusions.',
    tools=[],
    verbose=True,
    backstory=
    ("A proficient medical writer with expertise in synthesizing complex clinical information into clear and concise documents. "
     "You ensure that all findings are documented in a coherent diagnostic framework."
     ),
    llm=llm,  # Specify gpt-4o model here
    allow_delegation=False)

logging.info("Agents initialized successfully.")

# Define tasks with context and error handling
initial_assessment_task = Task(
    description=
    ("Collect information on the patient's chief complaint and history of present illness (HPI): {clinical_history}, {chief_complaint}. "
     "Document the findings based on user inputs and identify any immediate areas needing further investigation."
     ),
    expected_output=
    ("A detailed account of the patient's chief complaint and HPI: {clinical_history}, {chief_complaint}. "
     "Include a section for 'Immediate Findings' based on user inputs and 'Suggested Areas for Further Investigation' with rationales."
     ),
    agent=head_internal_medicine,
    context=[])

comprehensive_medical_history_task = Task(
    description=
    ("Gather comprehensive medical history, including past medical history, medications, allergies, family history, and social history for {clinical_history}, {chief_complaint}. "
     "Document the findings based on user inputs and identify any gaps or areas needing further investigation with rationales."
     ),
    expected_output=
    ("A comprehensive medical history of the patient: {clinical_history}, {chief_complaint}. "
     "Include sections for 'Documented Findings' based on user inputs and 'Areas Needing Further Investigation' with explanations for each suggestion."
     ),
    agent=semiology_expert,
    context=[initial_assessment_task])

review_of_systems_task = Task(
    description=
    ("Conduct a review of systems (ROS) to identify any additional symptoms across different body systems: {clinical_history}, {chief_complaint}. "
     "Document the findings based on user inputs and suggest additional symptoms to investigate, explaining the reasons based on the comprehensive medical history."
     ),
    expected_output=
    ("A detailed review of systems (ROS) for the patient: {clinical_history}, {chief_complaint}. "
     "Include sections for 'Documented Symptoms' based on user inputs and 'Suggested Symptoms for Further Investigation' with rationales for each suggestion."
     ),
    agent=semiology_expert,
    context=[comprehensive_medical_history_task])

physical_examination_task = Task(
    description=
    ("Perform a targeted physical examination based on the information gathered so far: {clinical_history}, {chief_complaint}. "
     "Document the findings based on user inputs and suggest additional areas to examine, providing explanations based on the current clinical context."
     ),
    expected_output=
    ("A detailed report of the physical examination findings: {clinical_history}, {chief_complaint}. "
     "Include sections for 'Documented Findings' based on user inputs and 'Suggested Areas for Further Examination' with explanations for each suggestion."
     ),
    agent=physical_exam_specialist,
    context=[review_of_systems_task])

differential_diagnosis_task = Task(
    description=
    ("Generate a list of possible diagnoses (differential diagnosis) based on the patient's symptoms, history, and physical examination findings: {clinical_history}, {chief_complaint}. "
     "Document the initial differential diagnosis and suggest additional diagnoses to consider, providing rationales based on the gathered information."
     ),
    expected_output=
    ("A prioritized list of possible diagnoses: {clinical_history}, {chief_complaint}. "
     "Include sections for 'Initial Differential Diagnosis' based on gathered information and 'Suggested Additional Diagnoses' with explanations for each suggestion."
     ),
    agent=chief_differential_diagnosis,
    context=[
        initial_assessment_task, comprehensive_medical_history_task,
        review_of_systems_task, physical_examination_task
    ])

bayesian_reasoning_task = Task(
    description=
    ("Apply Bayesian reasoning to refine the differential diagnosis using known and unknown baseline probabilities: {clinical_history}, {chief_complaint}. "
     "Document the refined differential diagnosis and provide a rationale for each probability adjustment."
     ),
    expected_output=
    ("A refined differential diagnosis using Bayesian reasoning: {clinical_history}, {chief_complaint}. "
     "Include sections for 'Refined Diagnoses' and 'Probability Adjustments' with explanations for each."
     ),
    agent=chief_differential_diagnosis,
    context=[differential_diagnosis_task])

diagnostic_testing_task = Task(
    description=
    ("Identify appropriate diagnostic tests to gather more objective data for {clinical_history}, {chief_complaint}. "
     "Explain the rationale for each test based on the refined differential diagnosis. "
     "Outline a follow-up plan based on potential test outcomes."),
    expected_output=
    ("A detailed plan for diagnostic testing: {clinical_history}, {chief_complaint}. "
     "Include sections for 'Proposed Tests', 'Rationale for Each Test', and 'Follow-Up Plan Based on Potential Outcomes'."
     ),
    agent=chief_differential_diagnosis,
    context=[bayesian_reasoning_task])

synthesize_diagnostic_framework_task = Task(
    description=
    ("Synthesize all gathered information into a comprehensive diagnostic framework for the patient's clinical history and chief complaint: {clinical_history}, {chief_complaint}. "
     "Document the integrated findings and highlight key clinical points, providing rationales for the diagnostic conclusions."
     ),
    expected_output=
    ("A well-structured diagnostic framework integrating all gathered information, including the top 5-10 clinical pearls: {clinical_history}, {chief_complaint}. "
     "Include sections for 'Integrated Findings' and 'Key Clinical Points' with rationales for each diagnostic conclusion."
     ),
    agent=clinical_doc_specialist,
    context=[
        initial_assessment_task, comprehensive_medical_history_task,
        review_of_systems_task, physical_examination_task,
        differential_diagnosis_task, bayesian_reasoning_task,
        diagnostic_testing_task
    ])

logging.info("Tasks defined successfully.")

# Create the crew
crew = Crew(agents=[
    head_internal_medicine, semiology_expert, physical_exam_specialist,
    chief_differential_diagnosis, clinical_doc_specialist
],
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
