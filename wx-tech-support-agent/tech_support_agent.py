#
# Copyright IBM Corp. 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# Author: Nguyen, Hung (Howie) Sy
#

print("\033[90m(Please wait. Loading...)\033[0m")

from langchain.agents import AgentExecutor, tool, create_react_agent
from langchain_core.prompts import PromptTemplate
import traceback 
import prompts
import sys
import time
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style

# Load environment variables from the file .env 
from dotenv import load_dotenv
load_dotenv()

sys.path.append("../common_libs") # not a good pratice but it's ok in this case
from watsonx import WatsonxClient

# In pratice, you should have a mechanism to keep a limit of messages in the chat_memory because of token limits
chat_memory: list[dict[str, str]] = []

def agent_streaming_print(text: str, delay=0.005):
    print("Agent:", end=' ', flush=True)
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print(f"")  

def convert_chat_messages_dict_to_string(messages_dict) -> str: 
    if messages_dict:
        text = "\n".join(f"- {role}: {message}" for entry in messages_dict for role, message in entry.items())
        return text
    else:
        return ""

# Declare tool(s)
@tool("default_action", return_direct=False)
def default_action(input: str):
    """This is the default action that the agent can take when three is no other action"""
    
    # a workaround to remove "\nObservation" that may be found at the end
    input = input.removesuffix("\nObservation")

    return ("Instruction for you the agent: Based on the last messages in the chat history, "
            f"think about a relevant response in response to this user input '{input}'. "
            "Do not take the next action or use another tool, instead provide the final answer/response.\n")

@tool("generate_a_clarifying_question", return_direct=False)
def generate_a_clarifying_question(input: str):
    """Generating a clarifying question to gather relevant information about the issue.
    This will allow the issue to be identified more clearly"""

    prompt_template= PromptTemplate(input_variables=["chat_history"],
                                    template=prompts.PROMPT_TEMPLATE__CLARIFYING_QUESTIONS)
    prompt = prompt_template.format(chat_history=chat_memory)
    questions = granite_llm.invoke(prompt)
    questions = questions.strip().removesuffix("```").removeprefix("```")

    response = ("The instruction for you, the agent: Refer to the JSON array below for clarifying questions that you can use to ask the user:"
                f"\n{questions}\n" 
                "Extract the questions from the JSON array and think about combining the chosen questions into a single response if needed, especially when they are intended to identify a specific product or the object of the problem/issue. "
                )

    # print("\033[90m(Debug: The generate_a_clarifying_question tool was invoked)\033[0m")

    return response

@tool("diagnosis_and_solution", return_direct=True)
def diagnosis_and_solution(input: str):
    """This is to perform the step Diagnosis and Solution Suggestion. 
    Analyze gathered information to hypothesize the root cause of the issue. 
    And then suggest a solution based on the diagnosis information. Provide step-by-step instructions for resolving the issue. 
    Additionally, ask the user to confirm if the issue is resolved."""
    prompt_template= PromptTemplate(input_variables=["chat_history"],
                                    template=prompts.PROMPT_TEMPLATE__DIAGNOSIS_SOLUTION)
    prompt = prompt_template.format(chat_history=chat_memory)
    response = granite_llm.invoke(prompt)

    # print("\033[90m(Debug: The diagnosis_and_solution tool was invoked)\033[0m")

    return response

@tool("escalate_to_human_support", return_direct=True)
def escalate_to_human_support(input: str):
    """
    Escalate the issue to the human support as the issue could not resolved by the AI agent/assistant.
    If the input is too long, make a summary as the input instead.
    """

    input = input.removesuffix("\nObservation")

    email_body=("Hello the support team,\n\n"
        "Please look into the issue as shown below.\n\n" 
        f"{input}\n\n"
        f"Below is the conversation with the user so far.\n\n{convert_chat_messages_dict_to_string(chat_memory)}\n\n"
        "Thank you & Regards,\n\nSent by AI Agent" 
        )

    print(f"\033[90m(Debug: The escalate_to_human_support tool was invoked. An email is being sent to the support team:\n\n{email_body})\033[0m")

    result = ("Sorry I could not help resolve the issue. I've just escalated the case to the human support team, "
              "and they will reach out to you soon.\nIf anything else, please open a new support session by type 'new session' "
              "- or type 'quit' to exit the program. Thank you!")
    return result



if __name__ == "__main__":

    print("\nRequest the models from WatsonX...")

    llama_llm = WatsonxClient.request_llm(
                                model_id="meta-llama/llama-3-1-70b-instruct",
                                decoding_method="greedy", 
                                temperature=0.8,
                                repetition_penalty=1,
                                stop_sequences=None)
    global granite_llm
    granite_llm = WatsonxClient.request_llm(model_id="ibm/granite-3-8b-instruct",
                                decoding_method ="greedy", 
                                temperature = 0.7, 
                                max_new_tokens = 512,
                                stop_sequences=["<|end_of_text|>"])

    is_requested_to_stop = False

    MAX_ITERATIONS = 8
    NUMBER_OF_RETRIES = 0
    AGENT_EXECUTION_VERBOSE = False

    # Tools
    tools = [default_action, 
             generate_a_clarifying_question, 
             diagnosis_and_solution, 
             escalate_to_human_support]

    ### Set up ReAct framework ###

    prompt_template_react = PromptTemplate.from_template(prompts.PROMPT_TEMPLATE__REACT)

    # Construct the ReAct agent. The LLM is used as a reasoning engine
    agent = create_react_agent(llama_llm, tools, prompt_template_react)

    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, 
                                tools=tools,  
                                verbose=AGENT_EXECUTION_VERBOSE, 
                                handle_parsing_errors=True, 
                                max_iterations = MAX_ITERATIONS,
                                return_intermediate_steps=True,
                                )

    ### The main conversation ### 
    print("\nStart the support session! (Type 'quit' to finish, 'new session' for a new support session)\n")

    # Greeting
    greeting =  "Hello! This is the techical support. How can I help you?"
    agent_streaming_print(greeting)
    chat_memory.append({"Agent": greeting})

    user_style = Style.from_dict({
        'prompt': "#5dade2",
        '': "#5dade2"
    })

    # The conversation loop
    while True:
        try:
            # Perception 
            user_input = prompt([('class:prompt', f'\nYou: ')], style=user_style).strip()
            if user_input == "":
                continue
            print("\033[0m", end='')

            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'new session':
                if chat_memory:
                    chat_memory.clear()
                print("\nStart the new support session!\n")
                continue

            chat_memory.append({"User": user_input})

            print(f"\nAgent: \033[90mPlease wait\033[0m", end=' ', flush=True)
            for i in range(NUMBER_OF_RETRIES + 1):
                print(f"\033[90m...\033[0m", end=' ', flush=True)

                agent_response = None
                
                result = {}
                try:
                    result = agent_executor.invoke({"user_input": user_input, "chat_history": chat_memory})
                    agent_response = result.get('output')
                except Exception as e:
                    result['error'] = f"{e}"
                    traceback.print_exc() 

                if agent_response is not None and "Agent stopped due to iteration limit" in agent_response:
                    del result['output']
                    agent_response = None
                    result['error'] = "Agent stopped due to iteration limit or time limit"

                # if no good result returned from the model, let it retry
                if result.get('error'):
                    if i >= NUMBER_OF_RETRIES:
                        intermediate_steps = result.get('intermediate_steps')
                        if intermediate_steps:
                            print(f"\033[90m...\033[0m", end=' ', flush=True)
                            last_step_response = intermediate_steps[-1][-1] 
                            agent_response = last_step_response 
                            break
                else:
                    break # it's very good if it can reach here, so no need for a retry
            # End of the Retry loop    

            # after all retries, if still no result:
            if agent_response:
                chat_memory.append({"Agent": agent_response})
            else:
                agent_response = f"Sorry, something was wrong due to an error ({result.get('error')}). Please try entering the input again..."

            print(" " * 100, end='\r')
            agent_streaming_print(agent_response)

        except Exception as error: # any error else, stop the program
            print(f"Something's wrong: {error}")
            traceback.print_exc() 
            break

    print("\nExisted the program.\n")
