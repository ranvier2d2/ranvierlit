import os
import streamlit as st
import nest_asyncio
import asyncio
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq

# Apply nest_asyncio to manage nested event loops
nest_asyncio.apply()

# Set page config once here
st.set_page_config(page_title='Ranvier - Kronika', page_icon='🧠')
st.title("Review Enfermedades por IA ✨ ")
st.write("Welcome to the Ranvier-Kronika AI Skill sites. Use the sidebar to navigate to different pages.")

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

# Retrieve the API key from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

if not groq_api_key:
    st.error("GROQ_API_KEY environment variable not set. Please set the GROQ_API_KEY environment variable.")
    st.stop()

# Set up the customization options
st.sidebar.title('Customization')
model = st.sidebar.selectbox(
    'Choose a model',
    ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it', 'llama3-70b-8192']
)

# Initialize the language model with Groq
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=model)

# Define agents with verbose mode and backstories
researcher = Agent(
    role='Investigador',
    goal='Recopilar información completa sobre {disease_name}',
    tools=[],
    verbose=True,
    backstory=(
        "Un experimentado investigador médico con enfoque en epidemiología y fisiopatología.\n"
        "Para investigar {disease_name}, recopila información sobre:\n"
        "1. Características clínicas clave: signos, síntomas, sistemas corporales afectados, curso de la enfermedad y pronóstico\n"
        "2. Epidemiología: incidencia, prevalencia, poblaciones de alto riesgo, factores de riesgo y causas\n"
        "3. Fisiopatología: mecanismos biológicos subyacentes, función orgánica alterada, factores genéticos y ambientales\n"
        "4. Estrategias diagnósticas: evaluación diagnóstica típica, hallazgos clave en la historia y el examen, pruebas de laboratorio y estudios de imagen, pruebas especializadas"
    ),
    llm=llm,
    allow_delegation=False
)

analyst = Agent(
    role='Analista',
    goal='Analizar y sintetizar datos recopilados sobre {disease_name}',
    tools=[],
    verbose=True,
    backstory=(
        "Un hábil analista de datos con experiencia en análisis de datos médicos y predicción de resultados.\n"
        "Al analizar información sobre {disease_name}:\n"
        "1. Evaluar enfoques de manejo: objetivos del tratamiento, terapias médicas y quirúrgicas, cuidado multidisciplinario\n"
        "2. Analizar complicaciones y seguimiento: principales complicaciones, planes de monitoreo y seguimiento, factores que influyen en los resultados\n"
        "3. Utilizar recursos de información de alta calidad: libros de texto médicos, artículos de revistas, guías, opiniones de expertos"
    ),
    llm=llm,
    allow_delegation=False
)

writer = Agent(
    role='Escritor',
    goal='Compilar hallazgos sobre {disease_name} en una revisión coherente',
    tools=[],
    verbose=True,
    backstory=(
        "Un escritor médico competente con habilidad para sintetizar información compleja en documentos claros y concisos.\n"
        "Para escribir una revisión completa sobre {disease_name}:\n"
        "1. Sintetizar información para proporcionar una imagen completa de la enfermedad\n"
        "2. Explicar cómo {disease_name} encaja en los diagnósticos diferenciales de síntomas comunes\n"
        "3. Discutir cómo se puede aplicar el conocimiento clínicamente para mejorar el razonamiento diagnóstico y la toma de decisiones\n"
        "4. Utilizar una organización clara con secciones sobre características clínicas, epidemiología, fisiopatología, diagnóstico, manejo y complicaciones"
    ),
    llm=llm,
    allow_delegation=False
)

# Define tasks
collect_clinical_features_task = Task(
    description='Recopilar información sobre los signos, síntomas y manifestaciones clínicas típicas de {disease_name}',
    expected_output='Una lista detallada de características clínicas y curso de la enfermedad de {disease_name}',
    agent=researcher,
    context=[]
)

determine_epidemiology_task = Task(
    description='Determinar la incidencia, prevalencia y factores de riesgo de {disease_name}',
    expected_output='Un resumen de datos epidemiológicos de {disease_name}',
    agent=researcher,
    context=[collect_clinical_features_task]
)

review_pathophysiology_task = Task(
    description='Revisar los mecanismos biológicos y factores que conducen a {disease_name}',
    expected_output='Una explicación detallada de la fisiopatología de {disease_name}',
    agent=researcher,
    context=[determine_epidemiology_task]
)

familiarize_diagnostic_workup_task = Task(
    description='Familiarizarse con la evaluación diagnóstica, hallazgos clave y pruebas especializadas para {disease_name}',
    expected_output='Una lista completa de estrategias diagnósticas para {disease_name}',
    agent=researcher,
    context=[review_pathophysiology_task]
)

review_management_approaches_task = Task(
    description='Revisar tratamientos médicos y quirúrgicos, y cuidados multidisciplinarios para {disease_name}',
    expected_output='Un resumen de enfoques de manejo para {disease_name}',
    agent=analyst,
    context=[familiarize_diagnostic_workup_task]
)

recognize_complications_task = Task(
    description='Reconocer complicaciones, monitoreo y planes de seguimiento para {disease_name}',
    expected_output='Una lista detallada de complicaciones y estrategias de seguimiento para {disease_name}',
    agent=analyst,
    context=[review_management_approaches_task]
)

synthesize_information_task = Task(
    description='Sintetizar toda la información recopilada sobre {disease_name} en una revisión completa',
    expected_output='Un documento de revisión bien estructurado que integre el conocimiento en el razonamiento clínico para {disease_name}, incluyendo los 5-10 puntos clínicos más importantes',
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
disease_name = st.text_input("Ingresa una enfermedad o sindrome:", "")

if st.button("Iniciar Review"):
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
