import os
import streamlit as st
import nest_asyncio
import asyncio
from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic

# Apply nest_asyncio to manage nested event loops
nest_asyncio.apply()

# Set page config once here
st.set_page_config(page_title='Project Idea Processor', page_icon='🧠')
st.title("AI Enhanced Code Project Idea Processor ✨ ")
st.write(
    "Welcome to the Project Idea Processor. Use the sidebar to navigate to different pages."
)


# Function to load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Load the CSS file (optional, if you have a CSS file)
load_css("styles.css")

with st.expander('About this app'):
    st.markdown('''
				**What can this app do?**
				This app allows users to process a project idea using CrewAI agents. The agents will parse the idea, generate questions, provide answers, and compile the results into a coherent document.

				**How to use the app?**
				1. Enter your project idea in the input field.
				2. Click the "Start Processing" button to begin the process.
				3. The results will be displayed once the processing is completed.
				''')

# Ensure there is an event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError as e:
    if "There is no current event loop in thread" in str(e):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Retrieve the API key from environment variables
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not anthropic_api_key:
    st.error(
        "ANTHROPIC_API_KEY environment variable not set. Please set the ANTHROPIC_API_KEY environment variable."
    )
    st.stop()

llm = ChatAnthropic(temperature=0,
                    model_name="claude-3-5-sonnet-20240620",
                    api_key=anthropic_api_key)

# Agent definitions
project_parser_agent = Agent(
    role='Project Parser',
    goal='Parse user input to extract project details',
    verbose=True,
    memory=False,
    backstory=
    "You are adept at understanding and breaking down project ideas into key components.",
    llm=llm,
    allow_delegation=False)

question_generator_agent = Agent(
    role='Question Generator',
    goal='Generate relevant questions about the project',
    verbose=True,
    memory=False,
    backstory=
    "You excel at asking the right questions to uncover deeper insights about any project.",
    llm=llm,
    allow_delegation=False)

answer_generator_agent = Agent(
    role='Answer Generator',
    goal='Generate possible answers for the questions',
    verbose=True,
    memory=False,
    backstory=
    "You provide detailed and thoughtful answers to the questions generated.",
    llm=llm,
    allow_delegation=False)

result_presenter_agent = Agent(
    role='Result Presenter',
    goal='Format the results into a markdown document',
    verbose=True,
    memory=False,
    backstory=
    "You have a talent for organizing information into clear and structured documents.",
    llm=llm,
    allow_delegation=False)

refinement_assistant_agent = Agent(
    role='Senior Project Manager',
    goal='Refine questions and answers based on your expert coding review',
    verbose=True,
    memory=False,
    backstory=
    "You help refine and improve the questions and answers based on your tech lead expertise, focusing in a detailed code review",
    llm=llm,
    allow_delegation=False)

# Task definitions
parse_user_input_task = Task(
    description=
    """Identify the project description, key terms, technologies mentioned, and goals or objectives.
The user input will be provided in the following format:
<userRequest>
{project_idea}
</userRequest>

Reply with the extracted information wrapped in the following XML tags: 
<projectDescription>Project description goes here</projectDescription>
<keyTerms>Key terms go here, separated by commas</keyTerms>
<technologies>Technologies mentioned go here, separated by commas</technologies>
<goals>Project goals or objectives go here, separated by semicolons</goals>

If any of the above information is not found in the user input, leave that tag empty. Do not include any other text, explanation, or formatting in your reply.""",
    expected_output="Parsed project details in XML format.",
    agent=project_parser_agent)

generate_questions_task = Task(
    description=
    """Consider the project description, key terms, technologies, and goals mentioned in the parsed input. Generate questions that cover various aspects of the project, such as:
- Main objectives
- Target user or audience
- Suitable technologies or programming languages
- Potential challenges or roadblocks
- Monetization or value proposition
- Code dependencies and critical functions in pseudocode
- Code examples and explanations for the project
- Useful libraries and frameworks that already exist and can be imported to the project to aid with development.

Aim to generate around 8-12 questions. Each question should be on a new line.""",
    expected_output="A set of relevant questions.",
    agent=question_generator_agent,
    context=[parse_user_input_task])

generate_answers_task = Task(
    description=
    "Generate detailed possible answers for each question about the project.",
    expected_output=
    "Generated detailed answers for the questions. Making sure to a section of Suggested Code Snippets at the end that may be useful for the defined project",
    agent=answer_generator_agent,
    context=[parse_user_input_task, generate_questions_task])

present_results_task = Task(
    description=
    "Format the generated questions and answers into a markdown document.",
    expected_output="Formatted thorough markdown document.",
    agent=result_presenter_agent,
    context=[
        parse_user_input_task, generate_questions_task, generate_answers_task
    ])

refine_results_task = Task(
    description=
    "Refine questions and answers that form the project based on your expertise.",
    expected_output=
    "Refined questions and answer presented in a thorough professional document result of the Project and Code Review",
    agent=refinement_assistant_agent,
    context=[
        parse_user_input_task, generate_questions_task, generate_answers_task,
        present_results_task
    ])

# Forming the crew with sequential process
crew = Crew(agents=[
    project_parser_agent, question_generator_agent, answer_generator_agent,
    result_presenter_agent, refinement_assistant_agent
],
            tasks=[
                parse_user_input_task, generate_questions_task,
                generate_answers_task, present_results_task,
                refine_results_task
            ],
            process=Process.sequential)

# Streamlit input
project_idea = st.text_input("Enter your project idea:", "")

if st.button("Start Processing"):
    if project_idea:
        st.write(f"Processing project idea: {project_idea}...")
        inputs = {
            "project_idea": project_idea,
        }
        try:
            with st.spinner('Running CrewAI tasks...'):
                result = crew.kickoff(inputs=inputs)

                st.success("Processing completed!")

                detailed_results = []
                for task in crew.tasks:
                    task_result = task.output
                    detailed_results.append({
                        "task": task.description,
                        "result": task_result
                    })

                # Store detailed results and processing result in session state
                st.session_state['detailed_results'] = detailed_results
                st.session_state['processing_result'] = result

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a project idea.")

# Show processing result
if 'processing_result' in st.session_state:
    st.write(st.session_state['processing_result'])

# Show detailed results in an expander
if 'detailed_results' in st.session_state:
    with st.expander("Show detailed results"):
        for detail in st.session_state['detailed_results']:
            st.write(f"**Task:** {detail['task']}")
            st.write(f"**Result:** {detail['result']}")
            st.write("---")
