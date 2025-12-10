%%capture
%pip install crewai==0.186.1
%pip install crewai-tools==0.71.0
%pip install langchain-community==0.3.29
%pip install langchain-huggingface==0.3.1
%pip install sentence-transformers==5.1.0

%%capture

from crewai import Agent, Task, Crew, Process
from crewai import LLM
from crewai_tools import PDFSearchTool, SerperDevTool

llm = LLM(
    model="watsonx/ibm/granite-3-3-8b-instruct",
    base_url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
)
import os
os.environ['SERPER_API_KEY'] = 'API_KEY'
web_search_tool = SerperDevTool()

Creating our PDF Search Tool:

import warnings
warnings.filterwarnings('ignore') #Keeps Jupyter Notebook clean (not part of functionality)

pdf_search_tool = PDFSearchTool(
    pdf="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/7vgNfis17dQfjHAiIKkBOg/The-Daily-Dish-FAQ.pdf",
    config=dict(
        embedder=dict(
            provider="huggingface",
            config=dict(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
        )
    )
)



Step 1.1: Create the Agent¶
We define an Inquiry Specialist Agent whose job is to answer questions. Notice that the tools parameter is a list containing the pdf_search_tool and web_search_tool.

agent_centric_agent = Agent(
    role="The Daily Dish Inquiry Specialist",
    goal="""Accurately answer customer questions about The Daily Dish restaurant. 
    You must decide whether to use the restaurant's FAQ PDF or a web search to find the best answer.""",
    backstory="""You are an AI assistant for 'The Daily Dish'.
    You have access to two tools: one for searching the restaurant's FAQ document and another for searching the web.
    Your job is to analyze the user's question and choose the most appropriate tool to find the information needed to provide a helpful response.""",
    tools=[pdf_search_tool, web_search_tool],
    verbose=True,
    allow_delegation=False,
    llm=llm
)

agent_centric_task = Task(
    description="Answer the following customer query: '{customer_query}'. "
                "Analyze the question and use the tools at your disposal (PDF search or web search) to find the most relevant information. "
                "Synthesize the findings into a clear and friendly response.",
    expected_output="A comprehensive and well-formatted answer to the customer's query.",
    agent=agent_centric_agent
)

Step 1.3: Assemble the Crew
Finally, we create the Crew. It's a simple setup with our one agent and one task.

!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/7vgNfis17dQfjHAiIKkBOg/The-Daily-Dish-FAQ.pdf

print("\nWelcome to The Daily Dish Chatbot!")
print("What would you like to know? (Type 'exit' to quit)")

while True: 
    user_input = input("\nYour question: ").lower()
    if user_input == 'exit':
        print("Thank you for chatting. Have a great day!")
        break
    
    if not user_input:
        print("Please type a question.")
        continue

    try:
        # Here we use our more advanced, task-centric crew
        result_agent_centric = agent_centric_crew.kickoff(inputs={'customer_query': user_input})
        print("\n--- The Daily Dish Assistant ---")
        print(result_agent_centric)
        print("--------------------------------")
    except Exception as e:
        print(f"An error occurred: {e}")








task_centric_agent = Agent(
    role="Customer Service Specialist",
    goal="Provide exceptional customer service by following a multi-step process to answer customer questions accurately.",
    backstory="""You are an AI assistant for 'The Daily Dish'.
    You are an expert at following instructions. You will be given a sequence of tasks to complete.
    For each task, you will be provided with the specific tool needed to accomplish it.
    Your job is to execute each task diligently and pass the results to the next step.""",
    tools=[], # The agent is not given any tools directly
    verbose=True,
    allow_delegation=False,
    llm=llm
)

 **Step 2.2: Define the Tasks with Specific Tools**

Here is the core of the new approach. We define two distinct tasks. For the first two, we use the `tools` parameter within the `Task` definition itself to assign a specific tool to that step.

- **`faq_search_task`**: Is exclusively paired with `pdf_search_tool`.
- **`response_drafting_task`**: Needs the output (context) from the previous tasks but requires no tools of its own.

faq_search_task = Task(
    description="Search the restaurant's FAQ PDF for information related to the customer's query: '{customer_query}'.",
    expected_output="A snippet of the most relevant information from the PDF, or a statement that the information was not found.",
    tools=[pdf_search_tool], # Tool assigned directly to the task
    agent=task_centric_agent
)

response_drafting_task = Task(
    description="Using the information gathered from the FAQ search, draft a friendly and comprehensive response to the customer's query: '{customer_query}'.",
    expected_output="The final, customer-facing response.",
    agent=task_centric_agent,
    context=[faq_search_task]
)

Step 2.3: Assemble the New Crew
We assemble our new crew, providing the single agent and the list of three tasks. The Process.sequential setting ensures the tasks run in the order we've listed them

task_centric_crew = Crew(
    agents=[task_centric_agent],
    tasks=[faq_search_task, response_drafting_task],
    process=Process.sequential,
    verbose=True
)

print("\nWelcome to The Daily Dish Chatbot!")
print("What would you like to know? (Type 'exit' to quit)")

while True: 
    user_input = input("\nYour question: ").lower()
    if user_input == 'exit':
        print("Thank you for chatting. Have a great day!")
        break
    
    if not user_input:
        print("Please type a question.")
        continue

    try:
        # Here we use our more advanced, task-centric crew
        result_task_centric = task_centric_crew.kickoff(inputs={'customer_query': user_input})
        print("\n--- The Daily Dish Assistant ---")
        print(result_task_centric)
        print("--------------------------------")
    except Exception as e:
        print(f"An error occurred: {e}")


from crewai.tools import tool
import re

@tool("Add Two Numbers Tool")
def add_numbers(data: str) -> int:
    """
    Extracts and adds integers from the input string.
    Example input: 'add 1 and 2' or '[1,2,3,4]'
    Output: sum of the numbers
    """
    # Find all integers in the string
    numbers = list(map(int, re.findall(r'-?\d+', data)))
    return sum(numbers)

from functools import reduce

@tool("Multiply Numbers Tool")
def multiply_numbers(data: str) -> int:
    """
    Extracts and multiplies integers from the input string.
    Example input: 'multiply 2 and 3' or '[2,3,4]'
    Output: the product of all numbers found
    """
    numbers = list(map(int, re.findall(r'-?\d+', data)))
    return reduce(lambda x, y: x * y, numbers, 1)

calculator_agent = Agent(
    role="Calculator",
    goal="Extracts, adds, or multiplies numbers when asked, using the Add Two Numbers and Multiply Numbers tools.",
    backstory="An expert at parsing numeric instructions and computing sums or products.",
    tools=[add_numbers, multiply_numbers],
    llm=llm,
    allow_delegation=False
)

calculation_task = Task(
    description="Extract numbers from '{numbers}' and either add or multiply them, depending on the natural-language instruction.",
    expected_output="An integer result (sum or product) based on the user’s request.",
    agent=calculator_agent
)

crew = Crew(
    agents=[calculator_agent],
    tasks=[calculation_task],
    # verbose=True #Uncomment this to see the steps taken to get the final answer
)

# Inputs for addition…
result = crew.kickoff(inputs={'numbers': 'please add 4, 5, and 6'})
print("Sum result:", result)

# Inputs for multiplication…
result = crew.kickoff(inputs={'numbers': 'multiply 7 and 8 also 9 dont forget 10'})
print("Product result:", result)



