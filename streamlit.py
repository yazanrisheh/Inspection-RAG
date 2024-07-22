from crewai import Agent, Crew, Process, Task
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import streamlit as st
import os
import tempfile

load_dotenv()

# LLM initialization
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.2)

# --- Agents ---
research_agent = Agent(
    role="Research Agent",
    goal="Search through the PDF to find relevant answers",
    allow_delegation=False,
    verbose=True,
    backstory=(
        """
        The research agent is adept at searching and 
        extracting data from documents, ensuring accurate and prompt responses.
        """
    ),
    tools=[],
    llm=llm
)

professional_writer_agent = Agent(
    role="Professional Writer",
    goal="Write professional emails based on the research agent's findings",
    allow_delegation=False,
    verbose=True,
    backstory=(
        """
        The professional writer agent has excellent writing skills and is able to craft 
        clear and concise emails based on the provided information.
        """
    ),
    tools=[],
    llm=llm
)

# --- Tasks ---
answer_customer_question_task = Task(
    description=(
        """
        Answer the customer's questions based on the home inspection PDF.
        The research agent will search through the PDF to find the relevant answers.
        Your final answer MUST be clear and accurate, based on the content of the home
        inspection PDF.

        Here is the customer's question:
        {customer_question}
        """
    ),
    expected_output="""
        Provide clear and accurate answers to the customer's questions based on 
        the content of the home inspection PDF.
        """,
    tools=[],
    agent=research_agent,
)

write_email_task = Task(
    description=(
        """
        - Write a professional email to a contractor based 
            on the research agent's findings.
        - The email should clearly state the issues found in the specified section 
            of the report and request a quote or action plan for fixing these issues.
        - Ensure the email is signed with the following details:
        
            Best regards, "\n"
            Saber Sinan
        """
    ),
    expected_output="""
        Write a clear and concise email that can be sent to a contractor to address the 
        issues found in the home inspection report.
        """,
    tools=[],
    agent=professional_writer_agent,
)

# --- Crew ---
crew = Crew(
    agents=[research_agent, professional_writer_agent],
    tasks=[answer_customer_question_task, write_email_task],
    process=Process.sequential,
)

# Streamlit integration
def main():
    st.title("Construction Document Inspection")

    with st.sidebar:
        uploaded_file = st.file_uploader(label="Upload your inspection report", type=["pdf"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name
        
        pdf_search_tool = PDFSearchTool(pdf=pdf_path)
        research_agent.tools = [pdf_search_tool]

        customer_question = st.chat_input("Your message")
        if customer_question:
            result = crew.kickoff(inputs={"customer_question": customer_question})
            st.markdown(result)

if __name__ == "__main__":
    main()
