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
from collections import deque
import threading

# Load environment variables from the file .env 
from dotenv import load_dotenv
load_dotenv()

sys.path.append("../common_libs") # not a good pratice but it's ok in this case
from watsonx import WatsonxClient

MAX_ITERATIONS = 8
NUMBER_OF_RETRIES = 0
AGENT_EXECUTION_VERBOSE = False

class ChatMemory:
    def __init__(self):
        # We should keep a limit of messages in the chat_memory because of token limits
        # In this example, simply keep the last n messages. But in pratice, consider some better ways
        self._chat_messages: deque[str] = deque(maxlen=50)

    def get_chat_messages(self) -> list[str]:
        return list(self._chat_messages)
    
    def to_string(self) -> str: 
        if self._chat_messages:
            text = "\n".join(f"- {role}: {message}" for entry in self._chat_messages for role, message in entry.items())
            return text
        else:
            return ""
    
    def _add_message(self, role: str, message: str):
        self._chat_messages.append({f"{role}": message})

    def add_user_message(self, message: str):
        self._add_message("user", message)
    
    def add_agent_message(self, message: str):
        self._add_message("agent", message)

class TechSupportAgent():

    # for the demo and simplicity purposes, there is only an instance of TechSupportAgent
    _instance = None
    _lock = threading.Lock()  

    def __new__(cls, *args, **kwargs):
        # Prevent direct instantiation 
        if cls._instance is not None:
            raise Exception("This class is a Singleton. Use get_instance() to access the instance.")
        return super().__new__(cls)

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._init()

    @classmethod
    def get_instance(cls):
        # Use double-checked locking to ensure lazy initialization and thread-safety
        if cls._instance is None:
            with cls._lock:  
                if cls._instance is None: 
                    cls._instance = super().__new__(cls)
                    cls._instance.__init__()
        return cls._instance

    def _init(self):
        print("\nRequest the models from WatsonX...")
        self.llama_llm = WatsonxClient.request_llm(
                                    model_id="meta-llama/llama-3-1-70b-instruct",
                                    decoding_method="greedy", 
                                    temperature=0.8,
                                    repetition_penalty=1,
                                    stop_sequences=None)
        self.granite_llm = WatsonxClient.request_llm(model_id="ibm/granite-3-8b-instruct",
                                    decoding_method ="greedy", 
                                    temperature = 0.7, 
                                    max_new_tokens = 1024,
                                    stop_sequences=["<|end_of_text|>"])

        # Set up LangChain's ReAct framework
        self._tools = [self.default_action, 
             self.generate_a_clarifying_question, 
             self.diagnosis_and_solution, 
             self.escalate_to_human_support]
        
        self._prompt_template_react = PromptTemplate.from_template(prompts.PROMPT_TEMPLATE__REACT)

        # Construct the ReAct agent. The LLM is used as a reasoning engine
        self._agent = create_react_agent(self.llama_llm, self._tools, self._prompt_template_react)

        # Create an agent executor by passing in the agent and tools
        self._agent_executor = AgentExecutor(agent=self._agent, 
                                    tools=self._tools,  
                                    verbose=AGENT_EXECUTION_VERBOSE, 
                                    handle_parsing_errors=True, 
                                    max_iterations = MAX_ITERATIONS,
                                    return_intermediate_steps=True,
                                    )

        self.chat_memory = ChatMemory()

    @staticmethod
    @tool("default_action", return_direct=False)
    def default_action(input: str):
        """This is the default action that the agent can take when three is no other action"""
        
        # a workaround to remove "\nObservation" that may be found at the end
        input = input.removesuffix("\nObservation")

        return ("[The instruction for the agent]: Based on the last messages in the chat history, "
                f"think about a relevant response in response to this user input '{input}'. "
                "Do not take the next action or use another tool, instead provide the final answer/response.\n")

    @staticmethod
    @tool("generate_a_clarifying_question", return_direct=False)
    def generate_a_clarifying_question(input: str):
        """Generating a clarifying question to gather relevant information about the issue.
        This will allow the issue to be identified more clearly"""
        support_agent = TechSupportAgent.get_instance()

        prompt_template= PromptTemplate(input_variables=["chat_history"],
                                        template=prompts.PROMPT_TEMPLATE__CLARIFYING_QUESTIONS)
        prompt = prompt_template.format(chat_history=support_agent.chat_memory.to_string())
        questions = support_agent.granite_llm.invoke(prompt)
        questions = questions.strip().removesuffix("```").removeprefix("```")

        response = ("[The instruction for the agent]: Refer to the JSON array below for clarifying questions that you can use to ask the user:"
                    f"\n{questions}\n" 
                    "Extract the questions from the JSON array and think about combining the chosen questions into a single response if needed, especially when they are intended to identify a specific product or the object of the problem/issue. "
                    )

        # print("\033[90m(Debug: The generate_a_clarifying_question tool was invoked)\033[0m")

        return response

    @staticmethod
    @tool("diagnosis_and_solution", return_direct=True)
    def diagnosis_and_solution(input: str):
        """This is to perform the step Diagnosis and Solution Suggestion. 
        Analyze gathered information to hypothesize the root cause of the issue. 
        And then suggest a solution based on the diagnosis information. Provide step-by-step instructions for resolving the issue. 
        Additionally, ask the user to confirm if the issue is resolved."""
        support_agent = TechSupportAgent.get_instance()

        prompt_template= PromptTemplate(input_variables=["chat_history"],
                                        template=prompts.PROMPT_TEMPLATE__DIAGNOSIS_SOLUTION)
        prompt = prompt_template.format(chat_history=support_agent.chat_memory.to_string())
        response = support_agent.granite_llm.invoke(prompt)

        # print("\033[90m(Debug: The diagnosis_and_solution tool was invoked)\033[0m")

        return response

    @staticmethod
    @tool("escalate_to_human_support", return_direct=True)
    def escalate_to_human_support(input: str):
        """
        Escalate the issue to the human support as the issue could not resolved by the AI agent/assistant.
        If the input is too long, make a summary as the input instead.
        """
        support_agent = TechSupportAgent.get_instance()

        input = input.removesuffix("\nObservation")

        email_body=("Hello the support team,\n\n"
            "Please look into the issue as shown below.\n\n" 
            f"{input}\n\n"
            f"Below is the conversation with the user so far.\n\n{support_agent.chat_memory.to_string()}\n\n"
            "Thank you & Regards,\n\nSent by AI Agent" 
            )

        # print(f"\033[90m(Debug: The escalate_to_human_support tool was invoked. An email is being sent to the support team:\n\n{email_body})\033[0m")
        print(f"\033[90m(Debug: The escalate_to_human_support tool was invoked. An email is being sent to the support team)\033[0m")

        result = ("I'm sorry I couldn't resolve the issue. I have escalated the case to the human support team via email, "
                 f"and they will contact you shortly. Below is the email:\n\n~~~\n{email_body}\n~~~\n\n"
                 "If anything else, please open a new support session. Thank you!")
        return result
        
    def greet_user(self, username):
        try:
            greeting = self.granite_llm.invoke("You are a helpful agent assisting the user with troubleshooting technical issues. "
                                                f"The user's name is {username}. You task is now to greet the user. "
                                                "At the same time, you also say something to offer your help, for example 'How can I help you?'")
        except:
            greeting =  f"Hello {username}! This is the techical support. How can I help you?"

        self.chat_memory.add_agent_message(greeting)
        return greeting

    def query(self, user_input):
        try:
            self.chat_memory.add_user_message(user_input)

            for i in range(NUMBER_OF_RETRIES + 1):
                print(f"\033[90m...\033[0m", end=' ', flush=True)
                agent_response = None
                
                result = {}
                try:
                    result = self._agent_executor.invoke({"user_input": user_input, 
                                                    "chat_history": self.chat_memory.to_string()})
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
                            # print(f"DEBUG last_step_response:\n{last_step_response}\n")
                            agent_response = self.granite_llm.invoke("<|start_of_role|>system<|end_of_role|>You are a helpful agent (assistant) assisting the user with troubleshooting technical issues. "
                                     f"Your task is now to generate a response using the following context or instructions:\n{last_step_response}"
                                     "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>")
                            break
                else:
                    break # it's very good if it can reach here, so no need for a retry
            # End of the Retry loop    

            # after all retries, if still no result:
            if agent_response:
                self.chat_memory.add_agent_message(agent_response)
            else:
                agent_response = f"Sorry, something was wrong due to an error ({result.get('error')}). Please try entering the input again..."

        except Exception as error:
            agent_response = f"Program error: Something's wrong: {error}"
            print(f"\n{agent_response}")
        finally:
            return agent_response

    def clear_memory(self):
        self.chat_memory = ChatMemory()


def agent_streaming_print(text: str, delay=0.005):
    print("Agent:", end=' ', flush=True)
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print(f"")  

if __name__ == "__main__":
    support_agent = TechSupportAgent.get_instance()

    is_requested_to_stop = False

    ### The main conversation ### 
    print("\nStart the support session! (Type 'quit' to finish, 'new session' for a new support session)\n")

    # Greeting
    greeting = support_agent.greet_user("Howie")
    agent_streaming_print(greeting)

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
                support_agent.clear_memory()
                print("\nStart the new support session!\n")
                continue

            print(f"\nAgent: \033[90mPlease wait\033[0m", end=' ', flush=True)
            agent_response = support_agent.query(user_input)
            print(" " * 100, end='\r')
            agent_streaming_print(agent_response)

        except Exception as error: # any error else, stop the program
            print(f"Something's wrong: {error}")
            traceback.print_exc() 
            break

    print("\nExisted the program.\n")
