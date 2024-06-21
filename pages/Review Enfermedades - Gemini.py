import os
import streamlit as st
import nest_asyncio
import asyncio
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from concurrent.futures import ThreadPoolExecutor
import base64
import time

# Apply nest_asyncio to manage nested event loops
nest_asyncio.apply()

# Set page config once here
st.set_page_config(page_title='Ranvier - Kronika', page_icon='🧠')
st.title("Review Enfermedades por IA ✨")
st.write(
    "Welcome to the Ranvier-Kronika AI Skill sites. Use the sidebar to navigate to different pages."
)


# Function to load CSS
def load_css(file_name):
	try:
		with open(file_name) as f:
			st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
	except FileNotFoundError:
		st.error(
		    "CSS file not found. Please ensure the 'styles.css' file is present.")


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

# Retrieve the API key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
	st.error(
	    "GOOGLE_API_KEY environment variable not set. Please set the GOOGLE_API_KEY environment variable."
	)
	st.stop()

# Set up the customization options
st.sidebar.title('Customization')

# Expander for explanations
with st.sidebar.expander('Explicaciones del Modelo y Parámetros'):
	st.markdown('''
		**Selección de Modelo:**
		- **Gemini 1.5 Flash:** Modelo más reciente de Google. Enfocado en balancear calidad de respuesta, mayor rapidez y menor precio.
		- **Gemini 1.5 Pro:** Un modelo premium ligeramente más lento, pero más potente y capaz de generar salidas detalladas y complejas.

		**Temperatura:**
		- Controla la creatividad del modelo. Valores más bajos hacen que la salida sea más determinista, mientras que valores más altos la hacen más creativa y variada.

		**Máximo de Tokens de Salida:**
		- Determina el número máximo de tokens (2-3 tokens son una palabra) que el modelo puede generar. Valores más altos permiten respuestas más largas.
		''')

# Model selection
model_option = st.sidebar.selectbox(
    'Elige un modelo',
    ['Gemini 1.5 Flash', 'Gemini 1.5 Pro'],
    index=0  # Default value set to 'Gemini Flash'
)

# Model parameters
temperature = st.sidebar.slider('Temperature', 0.0, 1.0, 0.0)
max_output_tokens = st.sidebar.slider('Max Tokens de Salida',
                                      min_value=2000,
                                      max_value=8192,
                                      value=8192)

# Initialize the language model based on the selected option
try:
	if model_option == 'Gemini 1.5 Flash':
		llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
		                             temperature=temperature,
		                             max_output_tokens=max_output_tokens)
	elif model_option == 'Gemini 1.5 Pro':
		llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-001",
		                             temperature=temperature,
		                             max_output_tokens=max_output_tokens)
	st.sidebar.success(f'Model initialized: {model_option}')
except Exception as e:
	st.sidebar.error(f'Error initializing model: {e}')

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
    llm=llm,
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
    'A well-structured document in Spanish. It must integrate key clinical points and knowledge into clinical reasoning for {disease_name}, presented in Markdown, in professional technical language and suggesting further evidence-based resources.',
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


# Function to run the CrewAI process asynchronously
async def run_crewai_process(inputs):
	loop = asyncio.get_running_loop()
	with ThreadPoolExecutor() as pool:
		result = await loop.run_in_executor(pool, crew.kickoff, inputs)
	return result


# Modify the run_and_store_result function
async def run_and_store_result(inputs):
	result = await run_crewai_process(inputs)
	st.session_state['task_result'] = result
	st.session_state['task_running'] = False
	st.session_state['task_completed'] = True


# Modify the start_process function
def start_process(disease_name):
	if 'task_running' not in st.session_state or not st.session_state[
	    'task_running']:
		st.session_state['task_running'] = True
		st.session_state['task_result'] = None
		st.session_state['task_completed'] = False
		inputs = {"disease_name": disease_name}

		loop = asyncio.get_event_loop()
		if loop.is_running():
			asyncio.create_task(run_and_store_result(inputs))
		else:
			loop.run_until_complete(run_and_store_result(inputs))

#add this new function to create a download link for the markdown file:
def get_binary_file_downloader_html(bin_file, file_label='File'):
	with open(bin_file, 'rb') as f:
			data = f.read()
	bin_str = base64.b64encode(data).decode()
	href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
	return href

# Streamlit input
disease_name = st.text_input("Ingresa una enfermedad o sindrome:", "")

if st.button("Iniciar Review"):
	if disease_name:
		start_process(disease_name)
	else:
		st.warning("Please enter a disease name.")

# Display task progress or results
if 'task_running' in st.session_state:
		if st.session_state['task_running']:
				st.write("Please wait while the task is running...")

				# Display task list
				for i, task in enumerate(crew.tasks):
						if i < st.session_state.get('current_task', 0):
								st.success(f"✅ {task.description}")
						elif i == st.session_state.get('current_task', 0):
								st.info(f"🔄 {task.description}")
						else:
								st.write(f"⏳ {task.description}")

				time.sleep(1)  # Add a small delay to prevent excessive reruns
				st.rerun()
		elif st.session_state.get('task_completed', False):
				st.success("Research completed!")
				result = st.session_state['task_result']
				st.write(result)

				# Add 'Copy to Clipboard' button
				if st.button('Copy to Clipboard'):
						st.code(result, language='markdown')
						st.info('Please use Ctrl+C (or Cmd+C on Mac) to copy the text above.')

				# Add 'Download as Markdown' button
				if st.button('Download as Markdown'):
						with open('result.md', 'w', encoding='utf-8') as f:
								f.write(result)
						st.markdown(get_binary_file_downloader_html('result.md', 'Result'), unsafe_allow_html=True)

				# Show detailed results in an expander
				with st.expander("Show detailed results"):
						for task in crew.tasks:
								st.write(f"**Task:** {task.description}")
								if hasattr(task, 'output') and task.output is not None:
										st.write(f"**Result:** {task.output}")
								else:
										st.write("**Result:** No output available for this task.")
								st.write("---")