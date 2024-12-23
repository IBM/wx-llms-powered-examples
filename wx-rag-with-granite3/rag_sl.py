#
# Copyright IBM Corp. 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# Author: Nguyen, Hung (Howie) Sy
#

from io import StringIO
import os
import time
from rag_agent import RagAgent
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

### main
if __name__ == "__main__":
    user_name = None
    vector_datastore = None

    ragAgent = RagAgent()

    st.set_page_config(page_title = "Wx-RAG", page_icon="🧙‍♂️", menu_items={'About': "### This is an example of RAG-based application using the LLM Granite 3.0"})
    st.markdown("### A question-answering application example based on RAG using Granite 3.0")
    st.caption("(Powered by IBM Granite via watsonx.ai, with a UI built using Streamlit)")

    with st.sidebar:
        st.markdown("**An example of RAG-based application using IBM Granite 3.0**")
        user_name = st.text_input("User name:",  key="user_name", value="Howie")
        if not user_name:
            st.warning('Please enter your user name.')
            st.stop()
        st.success('(Signed in)')
        st.divider()

    if "messages" not in st.session_state:
        greeting = None
        with st.spinner("..."):
            greeting = f"Hello {user_name}! I'm here to assist with any questions related to IBM products."
        st.session_state["messages"] = [{"role": "assistant", "content": greeting}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_input := st.chat_input():
    
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        response = "Sorry! Could not get the result. Something wrong"
        
        with st.spinner("..."):
            response, _ = ragAgent.query(user_input)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)