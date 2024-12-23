#
# Copyright IBM Corp. 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# Author: Nguyen, Hung (Howie) Sy
#

print("\033[90m(Loading watsonx client...)\033[0m")
from langchain_core.prompts import PromptTemplate
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from kb_retriever import KnowledgeBaseRetriever 
print("\033[90m(watsonx client loaded)\033[0m")

from rag_prompt_template import RAG_PROMPT_TEMPLATE
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
import time
from collections import deque

import sys
sys.path.append("../common_libs") # not a good pratice but it's ok in this case
from watsonx import WatsonxClient

# Load environment variables from the file .env 
from dotenv import load_dotenv
load_dotenv()

def agent_streaming_print(text: str, delay=0.005):
    print("\nAgent:", end=' ', flush=True)
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print(f"")  

USER_STYLE = Style.from_dict({'prompt': "#5dade2", '': "#5dade2"})

class ChatMemory:
    def __init__(self):
        # We should keep a limit of messages in the chat_memory because of token limits
        # In this example, simply keep the last 10 messages. But in pratice, consider some better ways
        self._chat_memory: deque[str] = deque(maxlen=10)

    def get_chat_messages(self) -> list[str]:
        return list(self._chat_memory)
    
    def to_multiple_lines_string(self) -> str:
        str_chat_memory = "\n".join(list(self._chat_memory))
        return str_chat_memory
    
    def _add_message(self, role: str, message: str):
        self._chat_memory.append(f"<|start_of_role|>{role}<|end_of_role|>{message}<|end_of_text|>")

    def add_user_message(self, message: str):
        self._add_message("user", message)
    
    def add_assistant_message(self, message: str):
        self._add_message("assistant", message)


class RagAgent:
    def __init__(self):
        self._chat_memory = ChatMemory()
        self._llm = WatsonxClient.request_llm(model_id="ibm/granite-3-8b-instruct",
                                decoding_method = DecodingMethods.GREEDY, 
                                temperature = 0.7, 
                                max_new_tokens = 1024,
                                stop_sequences=["<|end_of_text|>"])

        self._prompt_template = PromptTemplate(input_variables=["context", "chat_history", "query"], 
                                        template=RAG_PROMPT_TEMPLATE)
    def greet_user(self, user_name):
        greeting = f"Hello {user_name}! I'm here to assist with any questions related to IBM products."
        self._chat_memory.add_assistant_message(greeting)
        return greeting
    
    def query(self, user_query):
        """Return a tuple in which the first item is the response from the model, 
        the 2nd item is a list of relevant text notes"""
        technotes = KnowledgeBaseRetriever.technote_retriever.invoke(user_query, k=1)
        technote_context = []
        for technote in technotes:
            # print(f"Debug: Score: {technote.metadata['score'] }")
            if technote.metadata['score'] >= 0.7:
                del technote.metadata['content'] # don't need content as just a HTML version of 'text'
                product_name = technote.metadata["note_metadata"]["productName"]
                del technote.metadata["note_metadata"]["productName"]
                technote_context.append({"product name": product_name, "document": technote})

        final_prompt = self._prompt_template.format(context=technote_context,
                                                chat_history=self._chat_memory.to_multiple_lines_string(),
                                                query=user_query)
        response = self._llm.invoke(final_prompt)

        self._chat_memory.add_user_message(user_query)
        self._chat_memory.add_assistant_message(response)
        
        return response, technote_context
    
    def clear_memory(self):
        self._chat_memory = ChatMemory()

if __name__ == "__main__":
    ragAgent = RagAgent()

    ### The main conversation ### 
    print("\nStart the support session! (Type 'quit' to finish, 'new session' for a new support session)\n")

    # Greeting
    greeting = ragAgent.greet_user("Howie")
    agent_streaming_print(greeting)

    # The conversation loop
    while True:
        try:
            user_input = prompt([('class:prompt', f'\nYou: ')], style=USER_STYLE).strip()
            if user_input == "":
                continue

            if user_input.lower().strip() == 'quit':
                break
            elif user_input.lower().strip() == 'new session':
                ragAgent.clear_memory()
                print("\nStart the new chat session!\n")
                continue
                
            response, _ = ragAgent.query(user_input)
            agent_streaming_print(response)

        except Exception as error:
            print(f"\nSomething's wrong: {error}")
            break

    KnowledgeBaseRetriever.cleanup()
    print("\nExisted the program.\n")
