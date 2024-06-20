import os
import streamlit as st
import nest_asyncio
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import logging

# Configure logging
logging.basicConfig(
		format='%(asctime)s %(levelname)s %(message)s',
		level=logging.DEBUG,
		handlers=[logging.StreamHandler()]
)

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

# Retrieve the API key from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if not groq_api_key:
	st.error(
	    "GROQ_API_KEY environment variable not set. Please set the GROQ_API_KEY environment variable in Replit secrets."
	)
	st.stop()

# Set up the customization options
st.sidebar.title('Customization')
model = st.sidebar.selectbox(
    'Choose a model',
    ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it', 'llama3-70b-8192'])

# Initialize the language model with Groq
llm = ChatGroq(temperature=0,
               groq_api_key=os.getenv("GROQ_API_KEY"),
               model_name=model)

# Define agents with enhanced roles, backstories, and goals
history_taker = Agent(
		role='Senior Semiology Professor',
		goal=(
				'Provide guidelines on how to gather a comprehensive patient history for the chief complaint: {chief_complaint}. '
				'Offer detailed suggestions on specific areas to explore based on the clinical context.'
		),
		tools=[],
		verbose=True,
		backstory=(
				"You have spent over 20 years in clinical practice, honing your skills in eliciting detailed and accurate patient histories. "
				"Your meticulous approach ensures that no detail is overlooked, providing a solid foundation for diagnosis related to {chief_complaint}."
		),
		llm=llm,
		allow_delegation=False
)
logging.info(f"Agent initialized: {history_taker}")

examiner = Agent(
		role='Senior Semiology Professor',
		goal=(
				'Provide guidelines on how to perform a targeted physical examination for the chief complaint: {chief_complaint}. '
				'Offer detailed suggestions on specific areas to examine based on the clinical context.'
		),
		tools=[],
		verbose=True,
		backstory=(
				"With extensive experience in physical examination and diagnostics, you excel at identifying subtle physical signs that can be crucial for diagnosis. "
				"Your thorough and systematic approach ensures that all relevant physical findings related to the patient's chief complaint are captured."
		),
		llm=llm,
		allow_delegation=False
)
logging.info(f"Agent initialized: {examiner}")

diagnostician = Agent(
		role='Head of Internal Medicine Department',
		goal=(
				'Provide guidelines on how to develop a differential diagnosis and recommend diagnostic tests for the chief complaint: {chief_complaint}. '
				'Offer detailed suggestions based on patient history and physical examination findings.'
		),
		tools=[],
		verbose=True,
		backstory=(
				"A leading figure in the field of diagnostics, you specialize in synthesizing complex clinical information to generate accurate differential diagnoses. "
				"Your analytical skills and experience allow you to identify the most likely causes and necessary tests for the chief complaint."
		),
		llm=llm,
		allow_delegation=False
)
logging.info(f"Agent initialized: {diagnostician}")

# Define agents with enhanced roles, backstories, and goals
history_taker = Agent(
		role='Senior Semiology Professor',
		goal=(
				'Provide guidelines on how to gather a comprehensive patient history for the chief complaint: {chief_complaint}. '
				'Offer detailed suggestions on specific areas to explore based on the clinical context. Do not infer any details about the patient’s history.'
		),
		tools=[],
		verbose=True,
		backstory=(
				"You have spent over 20 years in clinical practice, honing your skills in eliciting detailed and accurate patient histories and semiological-diagnostical heuristics. "
				"Your meticulous approach ensures that no detail is overlooked, providing a solid foundation for diagnosis related to patient's chief health complaints."
		),
		llm=llm,
		allow_delegation=False
)
logging.info(f"Agent initialized: {history_taker}")

examiner = Agent(
		role='Senior Semiology Professor',
		goal=(
				'Provide guidelines on how to perform a targeted physical examination for the chief complaint: {chief_complaint}. '
				'Offer detailed suggestions on specific areas to examine based on the clinical context. Do not infer any findings from the examination.'
		),
		tools=[],
		verbose=True,
		backstory=(
				"With extensive experience in physical examination and diagnostics, you excel at identifying subtle physical signs that can be crucial for diagnosis. "
				"Your thorough and systematic approach ensures that all relevant physical findings related to the patient's chief complaint are captured."
		),
		llm=llm,
		allow_delegation=False
)
logging.info(f"Agent initialized: {examiner}")

diagnostician = Agent(
		role='Head of Internal Medicine Department',
		goal=(
				'Provide guidelines on how to develop a differential diagnosis and recommend diagnostic tests for the chief complaint: {chief_complaint}. '
				'Offer detailed suggestions based on patient history and physical examination findings. Do not infer or assume any specific diagnoses without supporting evidence.'
		),
		tools=[],
		verbose=True,
		backstory=(
				"A leading figure in the field of diagnostics, you specialize in synthesizing complex clinical information to generate accurate differential diagnoses. "
				"Your analytical skills and experience allow you to identify the most likely causes and necessary tests for the chief complaint."
		),
		llm=llm,
		allow_delegation=False
)
logging.info(f"Agent initialized: {diagnostician}")

# Define tasks with context and error handling
gather_history_task = Task(
		description=(
				"Provide guidelines on how to collect a comprehensive patient history based on chief complaint: {chief_complaint}. "
				"Identify areas needing further exploration based on the clinical context."
		),
		expected_output=(
				"Patient History Guidelines for {chief_complaint}:\n\n"
				"- **Suggested Further Exploration:**\n"
				"  - Questions or areas that require further investigation based on the initial findings and clinical context.\n"
				"- **Suggested Topics for Further Anamnesis:**\n"
				"  - Additional medical conditions, medications, allergies, family history aspects, social history elements, and review of systems symptoms that should be further explored.\n"
		),
		agent=history_taker,
		context=[]
)
logging.info(f"Task initialized: {gather_history_task}")

perform_examination_task = Task(
		description=(
				"Provide guidelines on how to create a targeted physical examination template based on the patient history: {chief_complaint}. "
				"Suggest additional areas to examine with clinical context justifications."
		),
		expected_output=(
				"Physical Examination Guidelines for {chief_complaint}:\n\n"
				"- **Suggested Further Examinations:**\n"
				"  - Areas to be further examined with justifications based on the clinical context.\n"
				"- **Suggested Detailed Examination Areas:**\n"
				"  - Additional areas in general appearance, vitals, and specific systems (neurological, musculoskeletal, cardiovascular, respiratory) that should be further explored.\n"
		),
		agent=examiner,
		context=[gather_history_task]
)
logging.info(f"Task initialized: {perform_examination_task}")

generate_differential_diagnosis_task = Task(
		description=(
				"Provide guidelines on how to generate a differential diagnosis based on the patient history and physical examination findings related to chief complaint: {chief_complaint}. "
				"Provide rationales and probabilities for suggested diagnoses."
		),
		expected_output=(
				"Differential Diagnosis Guidelines for {chief_complaint}:\n\n"
				"- **Suggested Diagnoses:**\n"
				"  - List of potential diagnoses with rationales based on the clinical context.\n"
				"- **Rationale for Each Diagnosis:**\n"
				"  - Explanations for why each suggested diagnosis is considered, including probabilities based on clinical findings.\n"
		),
		agent=diagnostician,
		context=[gather_history_task, perform_examination_task]
)
logging.info(f"Task initialized: {generate_differential_diagnosis_task}")

bayesian_reasoning_task = Task(
		description=(
				"Provide guidelines on how to refine the differential diagnosis using Bayesian reasoning for {chief_complaint}. "
				"Adjust probabilities based on baseline knowledge and current findings. Separate known data from probabilistic reasoning."
		),
		expected_output=(
				"Bayesian Analysis Guidelines for chief complaint: {chief_complaint}:\n\n"
				"- **Refined Diagnoses:**\n"
				"  - Diagnoses with adjusted probabilities.\n"
				"- **Rationale for Each Adjustment:**\n"
				"  - Explanations for probability adjustments based on clinical findings and Bayesian reasoning.\n"
		),
		agent=diagnostician,
		context=[generate_differential_diagnosis_task]
)
logging.info(f"Task initialized: {bayesian_reasoning_task}")

synthesize_diagnostic_framework_task = Task(
		description=(
				"Provide guidelines on how to integrate all gathered information and advanced medical reasoning into a comprehensive diagnostic framework for the patient history: {chief_complaint}. "
				"Highlight key clinical points and provide rationales for diagnostic conclusions."
		),
		expected_output=(
				"Diagnostic Framework Guidelines for {chief_complaint}:\n\n"
				"- **Integrated Reasoning and Guidance:**\n"
				"  - Comprehensive synthesis of the proposed diagnostic framework.\n"
				"- **Rationales for Diagnostic Conclusions:**\n"
				"  - Explanations for each diagnostic conclusion based on integrated data.\n"
		),
		agent=diagnostician,
		context=[
				gather_history_task, perform_examination_task,
				generate_differential_diagnosis_task, bayesian_reasoning_task
		]
)
logging.info(f"Task initialized: {synthesize_diagnostic_framework_task}")

logging.info("Tasks defined successfully.")

# Create the crew
crew = Crew(agents=[history_taker, examiner, diagnostician],
						tasks=[
								gather_history_task, perform_examination_task,
								generate_differential_diagnosis_task, bayesian_reasoning_task,
								synthesize_diagnostic_framework_task
						],
						process=Process.sequential)

logging.info("Crew initialized successfully.")

# Streamlit inputs
chief_complaint = st.text_input("Enter chief complaint:", "")

if st.button("Start Clinical Assessment"):
		if chief_complaint:
				st.write(f"Assessing clinical history and chief complaint...")
				logging.info("Starting clinical assessment with provided inputs.")
				inputs = {
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
										logging.debug(f"Task result for {task.description}: {task_result}")
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