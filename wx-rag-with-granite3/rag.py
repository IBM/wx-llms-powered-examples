#
# Copyright IBM Corp. 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# Author: Nguyen, Hung (Howie) Sy
#

print("\033[90m(Please wait. Loading...)\033[0m")

from langchain_core.prompts import PromptTemplate
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from kb_retriever import KnowledgeBaseRetriever 
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

if __name__ == "__main__":

    llm = WatsonxClient.request_llm(model_id="ibm/granite-3-8b-instruct",
                                decoding_method = DecodingMethods.GREEDY, 
                                temperature = 0.7, 
                                max_new_tokens = 1024,
                                stop_sequences=["<|end_of_text|>"])

    prompt_template = PromptTemplate(input_variables=["context", "chat_history", "query"], 
                                     template=RAG_PROMPT_TEMPLATE)

    ### The main conversation ### 
    print("\nStart the support session! (Type 'quit' to finish, 'new session' for a new support session)\n")

    chat_memory = ChatMemory()

    # Greeting
    greeting = "Hello! I'm an AI assistant to help with answering questions related to IBM Products"
    agent_streaming_print(greeting)
    chat_memory.add_assistant_message(greeting)

    # The conversation loop
    while True:
        try:
            user_input = prompt([('class:prompt', f'\nYou: ')], style=USER_STYLE).strip()
            if user_input == "":
                continue

            if user_input.lower().strip() == 'quit':
                break
            elif user_input.lower().strip() == 'new session':
                chat_memory = ChatMemory()
                print("\nStart the new chat session!\n")
                continue

            technotes = KnowledgeBaseRetriever.technote_retriever.invoke(user_input, k=1)

            technote_context = []
            for technote in technotes:
                # print(f"Debug: Score: {technote.metadata['score'] }")
                if technote.metadata['score'] >= 0.7:
                    del technote.metadata['content'] # don't need content as just a HTML version of 'text'
                    product_name = technote.metadata["note_metadata"]["productName"]
                    del technote.metadata["note_metadata"]["productName"]
                    technote_context.append({"product name": product_name, "document": technote})

            final_prompt = prompt_template.format(context=technote_context,
                                                chat_history=chat_memory.to_multiple_lines_string(),
                                                query=user_input)
            
            response = llm.invoke(final_prompt)

            agent_streaming_print(response)

            chat_memory.add_user_message(user_input)
            chat_memory.add_assistant_message(response)
        
        except Exception as error:
            print(f"Something's wrong: {error}")
            break

    KnowledgeBaseRetriever.cleanup()
    print("\nExisted the program.\n")
