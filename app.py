"""Streamlit interface for the synthetic ABTalks coverage-assistant demo."""

import os
import re
import secrets

import requests
import streamlit as st

from response_cards import render_response_extras


st.set_page_config(
    page_title="Coverage Intelligence | ABTalks AI",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Viewer mode excludes Streamlit's Deploy action; CSS below hides the menu.
st.set_option("client.toolbarMode", "viewer")

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
DEFAULT_MEMBER_ID = 101  # Synthetic demo user, not an authentication mechanism.

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
  :root { --ink:#dce6f0; --muted:#91a4b8; --accent:#5eead4; --surface:#101e2e; --line:#293a4d; }
  html,body,[class*="css"] { font-family: 'DM Sans', sans-serif; }
  [data-testid="stAppViewContainer"] { background: radial-gradient(ellipse 65% 36% at 50% 0%, #173c4a 0%, transparent 78%), #08131f; color: var(--ink); }
  [data-testid="stHeader"] { background: transparent; }
  /* Hide Streamlit's top-right Deploy and theme/settings controls. */
  .stAppDeployButton, [data-testid="stAppDeployButton"],
  #MainMenu, [data-testid="stMainMenu"],
  .stAppToolbar, [data-testid="stToolbar"] { display: none !important; }
  [data-testid="stSidebar"] { background: #0c1b2b; border-right: 1px solid #21364a; }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #a8b9c8; }
  .block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 7rem; }
  h1,h2,h3 { color: #f4f9fd !important; font-family: 'Manrope', sans-serif !important; }
  .brand-row { display:flex; align-items:center; gap:14px; margin-bottom:26px; padding: 0 0 19px; border-bottom: 1px solid #243448; }
  .brand-symbol { display:grid; place-items:center; width:42px; height:42px; border:1px solid #446979; border-radius:13px; background:linear-gradient(145deg,#1c4d56,#102938); color:#8ef6e2; font-size:27px; }
  .brand-title { font:800 19px 'Manrope',sans-serif; letter-spacing:-.5px; color:#f3fbff; }
  .brand-sub { color:#8da7bb; font-size:12px; margin-top:1px; letter-spacing:.6px; text-transform:uppercase; }
  .hero-panel { border:1px solid #2b4a57; background:linear-gradient(140deg,rgba(32,81,86,.55),rgba(13,31,46,.88) 61%); padding:40px 42px 38px; border-radius:20px; margin:0 0 26px; box-shadow:0 24px 55px rgba(0,0,0,.11); }
  .hero-kicker { color:#87e9d8; text-transform:uppercase; letter-spacing:2px; font-size:12px; font-weight:700; }
  .hero-title { color:#fff; font:800 clamp(27px,3.5vw,44px)/1.2 'Manrope',sans-serif; max-width:650px; letter-spacing:-1.7px; margin:16px 0 15px; }
  .hero-body { color:#adc3d2; font-size:15px; line-height:1.75; max-width:690px; }
  .eyebrow { color:#7999ad; font-size:11px; font-weight:700; letter-spacing:1.8px; text-transform:uppercase; margin:24px 0 10px; }
  .status-pill { padding:7px 12px; border:1px solid #315769; background:#0e3040; border-radius:100px; color:#91e6d5; font-size:12px; font-weight:600; display:inline-block; margin-top:22px; }
  .section-heading { font:700 16px 'Manrope',sans-serif; color:#e9f4f7; margin:12px 0 13px; }
  .prompt-note { font-size:12px; color:#7f9cae; margin:5px 0 15px; }
  .safety-note { color:#7c98a9; font-size:12px; line-height:1.6; margin-top:18px; }
  [data-testid="stChatMessage"] { background:#102132; border:1px solid #254052; border-radius:16px; margin:11px 0; padding:8px 12px; }
  [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { color:#dfeaf1; line-height:1.7; }
  [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { color:inherit; }
  [data-testid="stChatInput"] { border:1px solid #35596b !important; background:#14283a !important; border-radius:15px; }
  [data-testid="stChatInput"] textarea { color:#ebf6fa !important; }
  .stButton > button { border:1px solid #325061; background:#142839; color:#def0f5; border-radius:12px; min-height:47px; font-weight:600; transition:background .16s, border .16s; }
  .stButton > button:hover { border-color:#5ac9b8; background:#1c3945; color:white; }
  [data-testid="stSidebar"] .stButton > button { background:#1d605e; border:1px solid #448783; color:#f8ffff; }
  [data-testid="stSidebar"] .stButton > button:hover { background:#25736e; }
  [data-testid="stExpander"] { border:1px solid #2b4657; border-radius:12px; background:#0e1f30; }
  .sidebar-wordmark { font:800 20px/1.25 'Manrope',sans-serif; color:#f0f9fb; letter-spacing:-.8px; padding-top:2px; }
  .sidebar-desc { color:#8ba4b4; font-size:12px; line-height:1.7; margin:6px 0 18px; }
  .stat-ok { color:#71e9bc; } .stat-off { color:#ffaf9f; }
  .sidebar-rule { margin:22px 0 16px; border:0; border-top:1px solid #2b4054; }
  @media(max-width: 740px) { .block-container { padding:1.2rem 1rem 7rem; } .hero-panel{ padding:26px 23px; } .brand-row{ margin-bottom:18px; } }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=12, show_spinner=False)
def system_status(url: str):
    try:
        response = requests.get(f"{url}/system-status", timeout=(3, 5))
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        try:
            response = requests.get(f"{url}/health", timeout=(3, 4))
            response.raise_for_status()
            return {"status": "ok", "ollama_reachable": None,
                    "model": "qwen3:8b", "context_length": None}
        except (requests.RequestException, ValueError):
            return {"status": "offline", "ollama_reachable": False,
                    "model": "qwen3:8b", "context_length": None}



def display_assistant_answer(answer: str):
    """Render assistant Markdown without misinterpreting currency as LaTeX."""
    readable = re.sub(r"(?<!`)`([^`\n]+)`(?!`)", r"\1", answer)
    # Streamlit treats pairs of $ characters as LaTeX delimiters.
    # Escape currency amounts (e.g. $500), but preserve other Markdown.
    readable = re.sub(r"(?<!\\)\$(?=\d)", r"\\$", readable)
    st.markdown(readable)


def show_context_warning(warning):
    if not isinstance(warning, dict):
        return
    tokens = warning.get("tokens_used")
    window = warning.get("context_length")
    if not isinstance(tokens, int) or not isinstance(window, int):
        return
    qualifier = "Estimated" if warning.get("estimated") else "Reported"
    st.warning(
        f"**Ollama context pressure** · {warning.get('model', 'Model')} · "
        f"{qualifier} usage {tokens:,} tokens / {window:,}-token context. "
        "Earlier context may have been discarded. Consider increasing the "
        "model's context length in Ollama. This warning does not necessarily "
        "explain a request timeout."
    )


def show_request_error(response):
    try:
        detail = response.json().get("detail", {})
    except (ValueError, AttributeError):
        detail = {}
    if isinstance(detail, dict):
        message = detail.get("message", "The request could not be completed.")
        warning = detail.get("context_warning")
    else:
        message = str(detail)
        warning = None
    if response.status_code >= 500 and response.status_code != 504:
        message = "The assistant encountered a server error. See backend logs for details."
    elif response.status_code == 429:
        message = "Request limit reached. Please try again shortly."
    elif response.status_code == 504 and not warning:
        message = "Retrieval timed out. No context-window exhaustion was confirmed; see backend logs for the cause."
    st.error(message)
    show_context_warning(warning)


if "session_id" not in st.session_state:
    st.session_state.session_id = secrets.randbelow(1_000_000_000) + 1
if "messages" not in st.session_state:
    st.session_state.messages = []
if "member_id" not in st.session_state:
    st.session_state.member_id = DEFAULT_MEMBER_ID

status = system_status(BACKEND_URL)
with st.sidebar:
    st.markdown('<div class="sidebar-wordmark">◈ Cohort Insurance Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-desc">Coverage Intelligence<br>Local agentic retrieval · Synthetic insurance data</div>', unsafe_allow_html=True)
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.session_id = secrets.randbelow(1_000_000_000) + 1
        st.session_state.messages = []
        st.rerun()
    st.markdown('<hr class="sidebar-rule">', unsafe_allow_html=True)
    st.markdown('**System status**')
    online = status.get("status") == "ok"
    st.markdown(
        f'<div class="{"stat-ok" if online else "stat-off"}">'
        f'{"● Backend connected" if online else "● Backend unavailable"}</div>',
        unsafe_allow_html=True,
    )
    reach = status.get("ollama_reachable")
    if reach is False:
        st.caption("Ollama is currently unreachable.")
    elif reach is True:
        st.caption("Ollama is connected.")
    st.markdown('**Runtime**')
    st.caption(f'Model · {status.get("model") or "qwen3:8b"}')
    ctx = status.get("context_length")
    st.caption(f'Active context · {ctx:,} tokens' if isinstance(ctx, int)
               else 'Active context · not available')
    st.markdown('<hr class="sidebar-rule">', unsafe_allow_html=True)
    st.markdown('**Usage Instructions**')
    st.caption('Ask about synthetic insurance plans, claims, benefits, or policy procedures.')
    st.caption('Compare multiple records in one question, then ask follow-up questions.')
    st.markdown('<hr class="sidebar-rule">', unsafe_allow_html=True)
    st.caption('Informational demo only · Not insurance or medical advice.')

st.markdown(
    '''<div class="brand-row"><div class="brand-symbol">◈</div>
    <div><div class="brand-title">Coverage Intelligence</div>
    <div class="brand-sub">ABTalks AI / Coverage Assistant</div></div></div>''',
    unsafe_allow_html=True,
)

selected_prompt = None
if not st.session_state.messages:
    st.markdown(
        '''<div class="hero-panel"><div class="hero-kicker">INSURANCE INTELLIGENCE · POWERED BY LOCAL AI</div>
        <div class="hero-title">Clear answers, grounded in your coverage data.</div>
        <div class="hero-body">Compare health plans, investigate claim records and navigate
        policy questions. Every answer is built from retrieved evidence, with verification
        before the assistant responds.</div><div class="status-pill">◉ Evidence-first assistant</div></div>''',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-heading">Start with a question</div>', unsafe_allow_html=True)
    suggestions = [
        ("Compare insurance plans", "Compare the premiums and deductibles of plans P101 and P102."),
        ("Investigate a claim", "What is the status of claim C1003, and how can a denied claim be appealed?"),
        ("Explore coverage", "What plan is member 101 enrolled in, and what benefits does it provide?"),
    ]
    columns = st.columns(3, gap="medium")
    for column, (label, query) in zip(columns, suggestions):
        with column:
            if st.button(label + "  ↗", use_container_width=True, help=query):
                selected_prompt = query
            st.markdown(f'<div class="prompt-note">{query}</div>', unsafe_allow_html=True)
    st.markdown('<div class="safety-note">All plans and claims in this demo are synthetic.</div>', unsafe_allow_html=True)
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                display_assistant_answer(message["content"])
            else:
                st.markdown(message["content"])
            if message["role"] == "assistant":
                show_context_warning(message.get("context_warning"))
                render_response_extras(
                    chunk_ids=message.get("chunk_ids"),
                    tool_result=message.get("tool_result"),
                    tool_calls=message.get("tool_calls"),
                )

prompt = st.chat_input("Ask about plans, claims, or policy coverage...") or selected_prompt
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "session_id": st.session_state.session_id,
        "member_id": st.session_state.member_id,
        "message": prompt,
    }
    with st.chat_message("assistant"):
        loading = st.status("Retrieving and verifying evidence...", expanded=False)
        try:
            # The backend currently sends one complete SSE response, not a token stream.
            with requests.post(
                f"{BACKEND_URL}/chat",
                json=payload,
                headers={"X-Member-ID": str(st.session_state.member_id)},
                stream=True,
                timeout=(5, 145),
            ) as response:
                if response.status_code != 200:
                    loading.update(label="Request unsuccessful", state="error")
                    show_request_error(response)
                else:
                    import json
                    full_answer = ""
                    chunk_ids, tool_result, tool_calls, context_warning = [], None, [], None
                    got_event = False
                    for line in response.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        got_event = True
                        full_answer += event.get("answer") or ""
                        chunk_ids = event.get("chunk_ids") or chunk_ids
                        tool_result = event.get("tool_result")
                        tool_calls = event.get("tool_calls") or tool_calls
                        context_warning = event.get("context_warning") or context_warning
                    if not got_event or not full_answer.strip():
                        loading.update(label="Empty response", state="error")
                        st.error("No usable answer was returned. Review backend logs.")
                    else:
                        loading.update(label="Answer ready", state="complete")
                        loading.empty()
                        display_assistant_answer(full_answer)
                        show_context_warning(context_warning)
                        render_response_extras(chunk_ids=chunk_ids, tool_result=tool_result,
                                               tool_calls=tool_calls)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_answer,
                            "chunk_ids": chunk_ids,
                            "tool_result": tool_result,
                            "tool_calls": tool_calls,
                            "context_warning": context_warning,
                        })
                        st.rerun()
        except requests.exceptions.Timeout:
            loading.update(label="Request timed out", state="error")
            st.error("The request exceeded the response time limit. Check backend logs; "
                     "context-window exhaustion has not been confirmed.")
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError):
            loading.update(label="Connection interrupted", state="error")
            st.error("The connection to the backend was interrupted. Check that the server is running.")
        except (ValueError, KeyError):
            loading.update(label="Invalid backend response", state="error")
            st.error("The backend returned an unexpected response format. Check backend logs.")
