import os
import logging
import streamlit as st
import nest_asyncio
import asyncio
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    handlers=[logging.StreamHandler()])

# Apply nest_asyncio to manage nested event loops
nest_asyncio.apply()

# Set page config
st.set_page_config(page_title='Ranvier - Kronika', page_icon='🧠')
st.title("Revisión de Enfermedades por IA ✨ ")
st.write(
    "Bienvenido a los sitios de habilidades de IA de Ranvier-Kronika. Usa la barra lateral para navegar a diferentes páginas."
)


# Function to load CSS
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logging.info(f"Loaded CSS file: {file_name}")
    except FileNotFoundError:
        logging.error(f"CSS file {file_name} not found.")
        st.error(f"CSS file {file_name} not found.")


# Load the CSS file
load_css("styles.css")

with st.expander('Acerca de esta aplicación'):
    st.markdown('''
    **¿Qué puede hacer esta aplicación?**
    Esta aplicación permite a los usuarios iniciar un proceso de investigación integral sobre enfermedades específicas utilizando agentes de CrewAI. Los agentes recopilarán, analizarán y compilarán la información en una revisión coherente.

    **¿Cómo usar la aplicación?**
    1. Ingresa el nombre de una enfermedad en el campo de entrada.
    2. Haz clic en el botón "Iniciar Revisión" para comenzar el proceso.
    3. Los resultados se mostrarán una vez que la investigación esté completa.
    ''')
    logging.info("Displayed information about the application.")

# Ensure there is an event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError as e:
    if "There is no current event loop in thread" in str(e):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    logging.debug("Created a new event loop.")

# Retrieve the API key from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if not google_api_key:
    st.error(
        "GOOGLE_API_KEY environment variable not set. Please set the GOOGLE_API_KEY environment variable."
    )
    st.stop()

if not groq_api_key:
    st.error(
        "La variable de entorno GROQ_API_KEY no está configurada. Por favor, configura la variable de entorno GROQ_API_KEY en Replit secrets."
    )
    logging.error("GROQ_API_KEY environment variable not set.")
    st.stop()

logging.info("GROQ_API_KEY environment variable retrieved successfully.")

# Set up the customization options
st.sidebar.title('Personalización')
model = st.sidebar.selectbox(
    'Elige un modelo',
    ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it', 'llama3-70b-8192'])
logging.info(f"Model selected: {model}")

# Initialize the language model with Groq
llm = ChatGroq(
    temperature=0, 
    groq_api_key = os.getenv("GROQ_API_KEY"), 
    model_name=model
)
logging.info("Initialized language model with Groq.")

# Define agents with verbose mode and backstories
investigador = Agent(
    role='Epidemiólogo e Investigador Clínico',
    goal=
    'Realizar una investigación profunda sobre los aspectos clínicos y epidemiológicos de {disease_name}',
    tools=[],
    verbose=True,
    backstory=
    ("Un epidemiólogo experimentado con amplia experiencia en la investigación de enfermedades infecciosas y condiciones crónicas. "
     "Tu objetivo es recopilar datos clínicos completos, estadísticas epidemiológicas y conocimientos sobre {disease_name}. "
     "Te centrarás en: \n"
     "1. Características clínicas: síntomas, progresión, pronóstico.\n"
     "2. Epidemiología: incidencia, prevalencia, poblaciones de alto riesgo, factores de riesgo y causas.\n"
     "3. Diagnóstico: métodos diagnósticos, hallazgos clave, pruebas de laboratorio y estudios de imagen."
     ),
    llm=llm,
    allow_delegation=False)

analista = Agent(
    role='Analista de Tratamiento y Sintetizador de Datos',
    goal=
    'Analizar los datos recopilados y sintetizarlos en conocimientos prácticos sobre el manejo y los resultados de {disease_name}',
    tools=[],
    verbose=True,
    backstory=
    ("Un experto en análisis de datos médicos y predicción de resultados. Tu tarea es analizar las opciones de tratamiento, las complicaciones potenciales y "
     "las estrategias de seguimiento para {disease_name}. Te centrarás en: \n"
     "1. Enfoques de tratamiento: médicos, quirúrgicos, cuidado multidisciplinario.\n"
     "2. Complicaciones y seguimiento: complicaciones clave, planes de monitoreo, factores que influyen en los resultados.\n"
     "3. Recursos basados en evidencia: libros de texto médicos, artículos de revistas, guías y opiniones de expertos."
     ),
    llm=llm,
    allow_delegation=False)

escritor = Agent(
    role='Escritor Médico y Revisor Jefe',
    goal=
    'Compilar y estructurar los hallazgos en una revisión médica coherente y completa',
    tools=[],
    verbose=True,
    backstory=
    ("Un escritor médico competente con habilidad para sintetizar información médica compleja en documentos claros y concisos. "
     "Tu tarea es escribir una revisión detallada sobre {disease_name}, incorporando: \n"
     "1. Características clínicas y epidemiología.\n"
     "2. Fisiopatología y diagnóstico.\n"
     "3. Estrategias de manejo y complicaciones.\n"
     "4. Aplicaciones clínicas y ayudas para la toma de decisiones."),
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0),
    allow_delegation=False)

logging.info("Agents defined successfully.")

# Define tasks
recopilar_caracteristicas_clinicas = Task(
    description=
    'Recopilar información detallada sobre los signos, síntomas y manifestaciones clínicas de {disease_name}',
    expected_output=
    'Una lista completa de características clínicas y la progresión de {disease_name}',
    agent=investigador,
    context=[])

determinar_epidemiologia = Task(
    description=
    'Determinar la incidencia, prevalencia y factores de riesgo asociados con {disease_name}',
    expected_output=
    'Un resumen detallado de datos epidemiológicos de {disease_name}',
    agent=investigador,
    context=[recopilar_caracteristicas_clinicas])

revisar_fisiopatologia = Task(
    description=
    'Revisar los mecanismos biológicos y factores que conducen a {disease_name}',
    expected_output=
    'Una explicación detallada de la fisiopatología de {disease_name}',
    agent=investigador,
    context=[determinar_epidemiologia])

familiarizarse_evaluacion_diagnostica = Task(
    description=
    'Familiarizarse con los métodos diagnósticos, hallazgos clave y pruebas especializadas para {disease_name}',
    expected_output=
    'Una lista completa de estrategias diagnósticas para {disease_name}',
    agent=investigador,
    context=[revisar_fisiopatologia])

revisar_enfoques_manejo = Task(
    description=
    'Revisar los tratamientos médicos y quirúrgicos, y el cuidado multidisciplinario para {disease_name}',
    expected_output='Un resumen de enfoques de manejo para {disease_name}',
    agent=analista,
    context=[familiarizarse_evaluacion_diagnostica])

reconocer_complicaciones = Task(
    description=
    'Reconocer las complicaciones potenciales, planes de monitoreo y estrategias de seguimiento para {disease_name}',
    expected_output=
    'Una lista detallada de complicaciones y estrategias de seguimiento para {disease_name}',
    agent=analista,
    context=[revisar_enfoques_manejo])

sintetizar_informacion = Task(
    description=
    'Sintetizar toda la información recopilada en una revisión completa de {disease_name}',
    expected_output=
    'Un documento bien estructurado, en markdown, en español, que integre el conocimiento en el razonamiento clínico para {disease_name} y finalmente entregue 10 take-home messages o Perlas Clinicas para recordar.',
    agent=escritor,
    context=[
        recopilar_caracteristicas_clinicas, determinar_epidemiologia,
        revisar_fisiopatologia, familiarizarse_evaluacion_diagnostica,
        revisar_enfoques_manejo, reconocer_complicaciones
    ])

logging.info("Tasks defined successfully.")

# Crear el crew
crew = Crew(agents=[investigador, analista, escritor],
            tasks=[
                recopilar_caracteristicas_clinicas, determinar_epidemiologia,
                revisar_fisiopatologia, familiarizarse_evaluacion_diagnostica,
                revisar_enfoques_manejo, reconocer_complicaciones,
                sintetizar_informacion
            ],
            process=Process.sequential)

logging.info("Crew created successfully.")

# Entrada de Streamlit
disease_name = st.text_input("Ingresa una enfermedad o síndrome:", "")

if st.button("Iniciar Revisión"):
    if disease_name:
        st.write(f"Investigando {disease_name}...")
        inputs = {"disease_name": disease_name}
        try:
            with st.spinner('Ejecutando tareas de CrewAI...'):
                result = crew.kickoff(inputs=inputs)
                st.success("Investigación completada!")

                detailed_results = []
                for task in crew.tasks:
                    task_result = task.output
                    detailed_results.append({
                        "task": task.description,
                        "result": task_result
                    })

                # Guardar resultados detallados y resultado de investigación en el estado de la sesión
                st.session_state['detailed_results'] = detailed_results
                st.session_state['research_result'] = result
                logging.info("Research completed successfully.")
        except Exception as e:
            st.error(f"Ocurrió un error: {str(e)}")
            logging.error(f"Error during research: {str(e)}")
    else:
        st.warning("Por favor, ingresa el nombre de una enfermedad.")
        logging.warning("No disease name entered.")

# Mostrar resultado de investigación
if 'research_result' in st.session_state:
    st.write(st.session_state['research_result'])
    logging.info("Displayed research result.")

# Mostrar resultados detallados en un expander
if 'detailed_results' in st.session_state:
    with st.expander("Mostrar resultados detallados"):
        for detail in st.session_state['detailed_results']:
            st.write(f"**Tarea:** {detail['task']}")
            st.write(f"**Resultado:** {detail['result']}")
            st.write("---")
        logging.info("Displayed detailed results.")
