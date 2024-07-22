# Inspection-RAG

## Overview
Inspection-RAG is a tool designed to generate inspection reports for homes. It reads input data, processes it, and generates a report in PDF format. This can be useful for real estate inspections, quality checks, and other similar use cases.

## Features
- **Automated Data Processing**: Reads and processes input inspection report to generate email that would normally take few hours.


## Requirements
- Python 3.9+ (3.11.5 was used)
- Libraries listed in `requirements.txt`
- Any LLM (Large Language Model) can be used such as Groq, Anthropic, OpenAI, Gemini, etc.
- Environment variable for the LLM API key, e.g., `groq_api_key=XXX` for Groq.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yazanrisheh/Inspection-RAG.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Inspection-RAG
    ```
3. Create a virtual environment:
    ```sh
    python -m venv venv
    ```
4. Activate the virtual environment:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```
5. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

## Usage without Streamlit
1. Prepare your input data according to the required format.
2. Ensure the environment variable for the LLM API key is set in your `.env` file:
    ```env
    groq_api_key=XXX
    ```
3. Run the main script to generate the report:
    ```sh
    python main.py
    ```

## Usage with Streamlit
1. Ensure the environment variable for the LLM API key is set in your `.env` file:
    ```env
    groq_api_key=XXX
    ```
2. Run the Streamlit application:
    ```sh
    streamlit run streamlit.py
    ```
3. Upload the PDF document and enter the query in the Streamlit interface such as "electrical system" as shown in the page 3. You can use any of the content.

