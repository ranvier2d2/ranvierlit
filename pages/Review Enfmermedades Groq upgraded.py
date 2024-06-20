import os
import streamlit as st
import nest_asyncio
import asyncio
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Apply nest_asyncio to manage nested event loops
nest_asyncio.apply()

# Set page config once here
st.set_page_config(page_title='Ranvier - Kronika', page_icon='🧠')
st.title("Review Enfermedades por IA ✨ ")
st.write(
    "Welcome to the Ranvier-Kronika AI Skill sites. Use the sidebar to navigate to different pages."
)


# Function to load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Load the CSS file
load_css("styles.css")

with st.expander('Acerca de esta aplicación'):
    st.markdown('''
		**¿Qué puede hacer esta aplicación?**
		Esta aplicación permite a los usuarios iniciar un proceso de investigación integral sobre enfermedades específicas utilizando agentes de CrewAI. Los agentes recopilarán, analizarán y compilarán la información en una revisión coherente.

		**¿Cómo usar la aplicación?**
		1. Ingresa el nombre de una enfermedad en el campo de entrada.
		2. Haz clic en el botón "Iniciar Review" para comenzar el proceso.
		3. Los resultados se mostrarán una vez que la investigación esté completa.
		''')

# Ensure there is an event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError as e:
    if "There is no current event loop in thread" in str(e):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

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
researcher = Agent(
    role='Epidemiologist and Clinical Investigator',
    goal=
    'Conduct an in-depth investigation into the clinical and epidemiological aspects of {disease_name}',
    tools=[],
    verbose=True,
    backstory=
    ("A seasoned epidemiologist with extensive experience in investigating infectious diseases and chronic conditions. "
     "Your goal is to compile comprehensive clinical data, epidemiological statistics, and insights into {disease_name}. "
     "You will focus on: \n"
     "1. Clinical features: symptoms, progression, prognosis.\n"
     "2. Epidemiology: incidence, prevalence, high-risk populations, risk factors, and causes.\n"
     "3. Diagnosis: diagnostic methods, key findings, lab tests, and imaging studies."
     ),
    llm=llm,
    allow_delegation=False)

analyst = Agent(
    role='Treatment Analyst and Data Synthesizer',
    goal=
    'Analyze the collected data and synthesize it into actionable insights regarding the management and outcomes of {disease_name}',
    tools=[],
    verbose=True,
    backstory=
    ("An expert in medical data analysis and outcome prediction. Your task is to analyze treatment options, potential complications, "
     "and follow-up strategies for {disease_name}. Focus on: \n"
     "1. Treatment approaches: medical, surgical, multidisciplinary care.\n"
     "2. Complications and follow-up: key complications, monitoring plans, factors influencing outcomes.\n"
     "3. Evidence-based resources: medical textbooks, journal articles, international society guidelines, and expert opinions."
     ),
    llm=llm,
    allow_delegation=False)

writer = Agent(
    role='Medical Writer and Reviewer',
    goal=
    'Compile and structure the findings into a coherent and comprehensive medical review',
    tools=[],
    verbose=True,
    backstory=
    ("A proficient medical writer skilled in synthesizing complex medical information into clear and concise documents. "
     "Your task is to write a detailed review on {disease_name}, incorporating: \n"
     "1. Clinical features and epidemiology.\n"
     "2. Pathophysiology and diagnosis.\n"
     "3. Management strategies and complications.\n"
     "4. Clinical applications and decision-making aids."),
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, max_output_tokens=8192),
    allow_delegation=False)

# Define tasks
collect_clinical_features_task = Task(
    description=
    'Gather detailed information about the signs, symptoms, and clinical manifestations of {disease_name}',
    expected_output=
    'A comprehensive list of clinical features and the progression of {disease_name}',
    agent=researcher,
    context=[])

determine_epidemiology_task = Task(
    description=
    'Determine the incidence, prevalence, and risk factors associated with {disease_name}',
    expected_output=
    'A detailed summary of epidemiological data for {disease_name}',
    agent=researcher,
    context=[collect_clinical_features_task])

review_pathophysiology_task = Task(
    description=
    'Review the biological mechanisms and factors leading to {disease_name}',
    expected_output=
    'An in-depth explanation of the pathophysiology of {disease_name}',
    agent=researcher,
    context=[determine_epidemiology_task])

familiarize_diagnostic_workup_task = Task(
    description=
    'Familiarize yourself with diagnostic methods, key findings, and specialized tests for {disease_name}',
    expected_output=
    'A comprehensive list of diagnostic strategies for {disease_name}',
    agent=researcher,
    context=[review_pathophysiology_task])

review_management_approaches_task = Task(
    description=
    'Review evidence-based medical and surgical treatments and multidisciplinary care for {disease_name}',
    expected_output=
    'A summary of evidence-based treatment approaches for {disease_name}',
    agent=analyst,
    context=[familiarize_diagnostic_workup_task])

recognize_complications_task = Task(
    description=
    'Recognize potential complications, monitoring plans, and follow-up strategies for {disease_name}',
    expected_output=
    'A detailed list of complications and follow-up strategies for {disease_name}',
    agent=analyst,
    context=[review_management_approaches_task])

synthesize_information_task = Task(
    description=
    'Synthesize all collected information into a comprehensive review of {disease_name}',
    expected_output=
    'A well-structured document integrating key clinical points and knowledge into clinical reasoning for {disease_name}, presented in Markdown translated to Spanish',
    agent=writer,
    context=[
        collect_clinical_features_task, determine_epidemiology_task,
        review_pathophysiology_task, familiarize_diagnostic_workup_task,
        review_management_approaches_task, recognize_complications_task
    ])

# Create the crew
crew = Crew(agents=[researcher, analyst, writer],
            tasks=[
                collect_clinical_features_task, determine_epidemiology_task,
                review_pathophysiology_task,
                familiarize_diagnostic_workup_task,
                review_management_approaches_task,
                recognize_complications_task, synthesize_information_task
            ],
            process=Process.sequential)

# Streamlit input
disease_name = st.text_input("Ingresa una enfermedad o sindrome:", "")

if st.button("Iniciar Review"):
    if disease_name:
        st.write(f"Researching {disease_name}...")
        inputs = {"disease_name": disease_name}
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
