import os
import streamlit as st
import nest_asyncio
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

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

# Define agents with verbose mode and backstories
history_taker = Agent(
    role='Senior Semiology Professor',
    goal=
    'Gather a comprehensive patient history and identify areas needing further exploration based on the clinical context of {chief_complaint}.',
    tools=[],
    verbose=True,
    backstory=
    ("You have spent over 20 years in clinical practice, honing your skills in eliciting detailed and accurate patient histories. "
     "Your meticulous approach ensures that no detail is overlooked, providing a solid foundation for diagnosis related to {chief_complaint}."
     ),
    llm=llm,
    allow_delegation=False)

examiner = Agent(
    role='Senior Semiology Professor',
    goal=
    'Perform a targeted physical examination and suggest further examination areas based on initial findings and the clinical context of {chief_complaint}.',
    tools=[],
    verbose=True,
    backstory=
    ("With extensive experience in physical examination and diagnostics, you excel at identifying subtle physical signs that can be crucial for diagnosis. "
     "Your thorough and systematic approach ensures that all relevant physical findings related to {chief_complaint} are captured."
     ),
    llm=llm,
    allow_delegation=False)

diagnostician = Agent(
    role='Head of Internal Medicine Department',
    goal=
    'Develop a differential diagnosis and recommend diagnostic tests for {chief_complaint} based on patient history and physical examination findings.',
    tools=[],
    verbose=True,
    backstory=
    ("A leading figure in the field of diagnostics, you specialize in synthesizing complex clinical information to generate accurate differential diagnoses. "
     "Your analytical skills and experience allow you to identify the most likely causes and necessary tests for {chief_complaint}."
     ),
    llm=llm,
    allow_delegation=False)

# Define tasks with context and error handling
gather_history_task = Task(
    description=
    'Collect a comprehensive patient history based on chief complaint: {chief_complaint}. Document findings from user inputs and suggest areas needing further exploration based on the clinical context of {chief_complaint}.',
    expected_output=
    ("Patient History Report and Suggested Relevant Anamnestic Investigation:\n"
     "- **Documented Findings**: \n"
     "  - History of Present Illness (HPI): Onset, duration, quality, severity, location, and associated symptoms related to chief complaint.\n"
     "  - Past Medical History (PMH): Relevant past medical conditions that may impact chief complaint.\n"
     "  - Medications: All current medications and possible interactions or side effects relevant to the chief complaint.\n"
     "  - Allergies: Known allergies including medications, food, and environmental allergies.\n"
     "  - Family History: Relevant family medical history impacting {chief_complaint}.\n"
     "  - Social History: Occupation, smoking history, alcohol consumption, drug use, exercise habits, and living situation, as relevant to chief complaint.\n"
     "  - Review of Systems (ROS): Only symptoms directly related to {chief_complaint} in different body systems.\n"
     "- **Suggested Further Anamnesis**: \n"
     "  - Justifications for each suggestion based on the initial findings and clinical context of the chief complaint."
     ),
    agent=history_taker,
    context=[])

perform_examination_task = Task(
    description=
    'Create a targeted physical examination template based on the patient history: {chief_complaint}. Document findings and suggest additional areas to examine with clinical context justifications related to this particular patient.',
    expected_output=
    ("Physical Examination Report for {chief_complaint}:\n"
     "- **Documented Findings**: \n"
     "  - General appearance and vitals\n"
     "  - Detailed findings for relevant systems (neurological, musculoskeletal, cardiovascular, etc.)\n"
     "- **Suggested Further Examinations**: \n"
     "  - Explanation for each suggested examination based on clinical context and initial findings related to {chief_complaint}."
     ),
    agent=examiner,
    context=[gather_history_task])

generate_differential_diagnosis_task = Task(
    description=
    'Generate a differential diagnosis based on the patient history and physical examination findings related to chief complaint: {chief_complaint}. Document initial diagnoses and suggest further diagnostic tests with explanations.',
    expected_output=
    ("Differential Diagnosis Report for {chief_complaint}:\n"
     "- **Initial Diagnoses**: \n"
     "  - Prioritized list with rationales based on chief complaint.\n"
     "- **Suggested Further Diagnostic Tests**: \n"
     "  - Rationale for each test based on clinical context and findings related to chief complaint."
     ),
    agent=diagnostician,
    context=[gather_history_task, perform_examination_task])

bayesian_reasoning_task = Task(
    description=
    'Refine the differential diagnosis using Bayesian reasoning for {chief_complaint}. Adjust probabilities based on baseline knowledge and current findings. Document refined diagnoses and suggest further diagnostic considerations.',
    expected_output=
    ("Bayesian Analysis Report for chief complaint: {chief_complaint}:\n"
     "- **Refined Diagnoses**: \n"
     "  - Adjusted probabilities with explanations based on chief complaint.\n"
     "- **Suggested Diagnostic Considerations**: \n"
     "  - Justifications based on Bayesian reasoning and clinical context of {chief_complaint}."
     ),
    agent=diagnostician,
    context=[generate_differential_diagnosis_task])

synthesize_diagnostic_framework_task = Task(
    description=
    'Integrate all gathered information into a comprehensive diagnostic framework for {chief_complaint}. Document key findings and highlight clinical points with rationales for diagnostic conclusions.',
    expected_output=
    ("Diagnostic Framework for the chief complaint:\n"
     "- **Integrated Findings**: \n"
     "  - Synthesis of all findings with key clinical points related to the chief complaint.\n"
     "- **Rationales for Diagnostic Conclusions**: \n"
     "  - Justifications based on integrated data for this chief complaint."),
    agent=diagnostician,
    context=[
        gather_history_task, perform_examination_task,
        generate_differential_diagnosis_task, bayesian_reasoning_task
    ])

# Create the crew
crew = Crew(agents=[history_taker, examiner, diagnostician],
            tasks=[
                gather_history_task, perform_examination_task,
                generate_differential_diagnosis_task, bayesian_reasoning_task,
                synthesize_diagnostic_framework_task
            ],
            process=Process.sequential)

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
