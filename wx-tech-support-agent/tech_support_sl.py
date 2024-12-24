#
# Copyright IBM Corp. 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# Author: Nguyen, Hung (Howie) Sy
#

from tech_support_agent import TechSupportAgent
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

def new_session():
    del st.session_state["messages"]    
    TechSupportAgent.get_instance().clear_memory()

if __name__ == "__main__":
    st.set_page_config(page_title = "Tech Support Agent", page_icon="🧙‍♂️", menu_items={'About': "### This example demonstrates an AI Agent designed to automate a technical support use case"})
    st.markdown("### This is an example of an AI agent designed to automate a technical support use case 💻🛠️")
    st.caption("(Powered by watsonx.ai, with a UI built using Streamlit)")

    if "support_agent" not in st.session_state:
         st.session_state["support_agent"] = TechSupportAgent.get_instance()
         TechSupportAgent.get_instance().clear_memory()

    support_agent: TechSupportAgent = st.session_state.support_agent

    if "user_name" not in st.session_state:
         st.session_state["user_name"] = "Howie"

    with st.sidebar:
        st.markdown("**🤖 Tech Support Agent**")
        st.text_input("User name:", key="user_name", 
                       on_change=new_session)
        if not st.session_state.user_name.strip():
            st.warning('Please enter your user name.')
            st.stop()
        st.success(f"(Signed in as {st.session_state.user_name} )")
        st.divider()
    
    user_avatar = "👨🏻"
    agent_avatar = "🤖"

    user_name = st.session_state.user_name

    if "messages" not in st.session_state:
        greeting = support_agent.greet_user(user_name)
        st.session_state["messages"] = [{"role": "assistant", "content": greeting}]

    for msg in st.session_state.messages:
        avatar = user_avatar if msg["role"] == "user" else agent_avatar
        st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    if user_input := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user", avatar=user_avatar).write(user_input)
        print(f"\nUser: {user_input}")
        
        with st.spinner("..."): 
            print(f"\nAgent: \033[90mPlease wait\033[0m", end=' ', flush=True)
            response = support_agent.query(user_input)
            print(" " * 100, end='\r')
            print(f"Agent: {response}")

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant", avatar=agent_avatar).write(response)