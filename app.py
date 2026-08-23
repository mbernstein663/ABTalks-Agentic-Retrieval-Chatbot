"""
Streamlit Frontend Development: Simple

Streamlt is fast and python-only while React is slower frontend engineering
Streamlit it aslo better for demos and model testing with moderate UI customiztion. Solid state handling and production featurs.


"""

import streamlit as st
import requests

import json
import sqlite3
import secrets
from pathlib import Path
from response_cards import render_response_extras




db_path = Path(__file__).resolve().parent / "data" / "coverage.db"

with sqlite3.connect(db_path) as conn:
    plans = conn.execute(
        "SELECT plan_id, plan_name FROM plans"
    ).fetchall()




url = "http://backend:8000/chat"

st.title("RAG Coverage Chatbot")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = secrets.randbelow(1_000_000_000) + 1

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello. I am here to provide informational assistant with claims, plans, or healthcare coverage. Ask away!"
        }
    ]

if "member_id" not in st.session_state:
    st.session_state.member_id = 101

# render the full conversation thread
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        render_response_extras(
            chunk_ids=message.get("chunk_ids"),
            tool_result=message.get("tool_result")
        )

# accept a new user message
if prompt := st.chat_input("What can I help with?"):
    user_message = {
        "role": "user",
        "content": prompt,
    }

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.write(prompt)



    
    payload = {
        "session_id": st.session_state["session_id"],
        "member_id": st.session_state["member_id"],
        "message": prompt
    }

    try:
        with st.chat_message("assistant"):
            status = st.status(
                "Generating response...",
                expanded=False
            )

            placeholder = st.empty()

            response = requests.post(
                url,
                json=payload,
                headers={"X-Member-ID": str(st.session_state["member_id"])},
                stream=True,
                timeout=(5, 120)
            )

            if response.status_code != 200:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text

                st.error(f"Backend error: {detail}")
                st.stop()

            if response.status_code == 200:
                full_answer = ""
                first_token = True

                tool_result = None
                answer_chunk_ids = []

                for line in response.iter_lines(
                    chunk_size=1,
                    decode_unicode=True
                ):
                    if not line or not line.startswith("data: "):
                        continue

                    response_data = json.loads(
                        line.removeprefix("data: ")
                    )

                    token = response_data.get("answer", "")

                    received_chunk_ids = response_data.get("chunk_ids", [])

                    if received_chunk_ids:
                        answer_chunk_ids = received_chunk_ids

                    if first_token:
                        status.update(
                            label="Response generated",
                            state="complete"
                        )
                        status.empty()
                        first_token = False

                    tool_result = response_data.get("tool_result")

                    full_answer += token
                    placeholder.markdown(full_answer)

                render_response_extras(
                    chunk_ids=answer_chunk_ids,
                    tool_result=tool_result
)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_answer,
                        "chunk_ids": answer_chunk_ids,
                        "tool_result": tool_result
                    }
                )

            else:
                status.update(
                    label="Response failed",
                    state="error"
                )

                st.error(
                    f"Error {response.status_code}: "
                    f"{response.json().get('detail')}"
                )

    except requests.exceptions.Timeout:
        status.empty()
        st.error(
            "The response timed out. Please try again."
        )

    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError
    ):
        status.empty()

        st.error(
            "The connection was interrupted while the response was streaming. "
            "Please try again."
        )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI server. "
            "Please check that it is running."
        )



with st.sidebar:
    st.header("Conversation Settings")

    selected_plan = st.selectbox(
        "Select a plan",
        options=plans,
        format_func=lambda plan: f"{plan[1]} ({plan[0]})"
    )

    selected_plan_id = selected_plan[0]

    if st.button("New conversation"):
        st.session_state.session_id = secrets.randbelow(1_000_000_000) + 1

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello. I can provide informational assistance with "
                    "claims, plans, or healthcare coverage. Ask away!"
                )
            }
        ]

        st.rerun()