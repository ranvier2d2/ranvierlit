import os
import streamlit as st
import nest_asyncio
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# Apply nest_asyncio to manage nested event loops
nest_asyncio.apply()

# Set page config
st.set_page_config(page_title='Medical Assistant - Page 2', page_icon='🩺')

st.title('🩺 Medical Assistant')

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
history_taker = Agent(
    role='History Taker',
    goal='Gather a comprehensive patient history based on the given chief complaint',
    tools=[],
    verbose=True,
    backstory=(
        "You are a seasoned clinician with a keen eye for detail, "
        "specializing in obtaining thorough patient histories. Your goal is to "
        "collect all relevant information to form a clear picture of the patient's condition. "
        "When taking a history for {chief_complaint}, you will:\n"
        "1. Inquire about onset, duration, progression, quality, severity, location, aggravating/relieving factors, and associated symptoms related to the chief complaint.\n"
        "2. Gather past medical history, medications, allergies, family history, and social history as relevant to the chief complaint.\n"
        "3. Conduct a review of systems to identify potential symptoms across different body systems, but only report findings directly related to the chief complaint."
    ),
    llm=llm,
    allow_delegation=False
)

examiner = Agent(
    role='Physical Examiner',
    goal='Perform a targeted physical examination based on the given chief complaint and patient history',
    tools=[],
    verbose=True,
    backstory=(
        "You are an expert clinician known for your thorough and precise physical examinations. "
        "Your role is to perform a detailed physical examination that complements the patient history "
        "for {chief_complaint}. In your examination, you will:\n"
        "1. Focus on relevant body systems based on the history and chief complaint.\n"
        "2. Look for signs that can help narrow down the differential diagnosis, strictly within the context of the chief complaint.\n"
        "3. Document findings meticulously to provide a clear basis for further analysis."
    ),
    llm=llm,
    allow_delegation=False
)

diagnostician = Agent(
    role='Diagnostician',
    goal='Generate a differential diagnosis and recommend diagnostic tests for the given chief complaint',
    tools=[],
    verbose=True,
    backstory=(
        "As a highly skilled diagnostician, your expertise lies in analyzing complex cases and generating accurate "
        "differential diagnoses. Your objective for {chief_complaint} is to:\n"
        "1. Compile a list of possible diagnoses based on the patient history and physical examination findings related to the chief complaint.\n"
        "2. Prioritize these diagnoses based on likelihood, severity, and urgency.\n"
        "3. Apply logical reasoning to refine probabilities and guide the selection of diagnostic tests, strictly within the context of the chief complaint."
    ),
    llm=llm,
    allow_delegation=False
)

# Define tasks with context and error handling
gather_history_task = Task(
    description='Gather a comprehensive patient history based on the given chief complaint. Focus on onset, duration, quality, severity, location, and associated symptoms. Also collect past medical history, medications, allergies, family history, and social history as relevant to the chief complaint.',
    expected_output=(
        "A detailed patient history including:\n"
        "- History of Present Illness (HPI): Onset, duration, quality, severity, location, and associated symptoms strictly related to the chief complaint.\n"
        "- Past Medical History (PMH): Relevant past medical conditions that may impact the chief complaint.\n"
        "- Medications: All current medications, including dosage and frequency, relevant to the chief complaint.\n"
        "- Allergies: Known allergies including medications, food, and environmental allergies.\n"
        "- Family History: Relevant family medical history impacting the chief complaint.\n"
        "- Social History: Occupation, smoking history, alcohol consumption, drug use, exercise habits, and living situation, as relevant to the chief complaint.\n"
        "- Review of Systems (ROS): Only symptoms directly related to the chief complaint in different body systems."
    ),
    agent=history_taker,
    context=[]
)

perform_examination_task = Task(
    description='Perform a targeted physical examination based on the given chief complaint and patient history. Focus on relevant body systems and document findings strictly related to the chief complaint.',
    expected_output=(
        "A comprehensive physical examination report including:\n"
        "- General: Patient's general appearance and distress level.\n"
        "- Vitals: Temperature, pulse, blood pressure, respiratory rate, and oxygen saturation.\n"
        "- Neurological: Mental status, cranial nerves, motor function, sensory function, reflexes, and gait strictly related to the chief complaint.\n"
        "- Musculoskeletal: Inspection, palpation, range of motion, and special tests (e.g., Straight Leg Raise Test) relevant to the chief complaint.\n"
        "- Cardiovascular: Auscultation of heart sounds and peripheral pulses relevant to the chief complaint.\n"
        "- Gastrointestinal: Abdominal inspection, auscultation, and palpation relevant to the chief complaint.\n"
        "- Genitourinary: Urinary symptoms and findings relevant to the chief complaint.\n"
        "- Skin: Inspection of the skin for rashes, erythema, or edema related to the chief complaint.\n"
        "- Respiratory: Auscultation of lung sounds relevant to the chief complaint.\n"
        "- Documentation: Thorough documentation of findings and patient's response to examination maneuvers strictly related to the chief complaint."
    ),
    agent=examiner,
    context=[gather_history_task]
)

generate_differential_diagnosis_task = Task(
    description='Generate a differential diagnosis and recommend diagnostic tests for the given chief complaint based on patient history and physical examination findings.',
    expected_output=(
        "A prioritized list of potential diagnoses including:\n"
        "- A brief description of each potential diagnosis strictly related to the chief complaint.\n"
        "- Reasoning for prioritizing each diagnosis based on likelihood, severity, and urgency.\n"
        "- Recommended diagnostic tests to confirm or rule out each potential diagnosis relevant to the chief complaint.\n"
        "- Application of logical reasoning to refine probabilities and guide diagnostic testing based on the chief complaint."
    ),
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
