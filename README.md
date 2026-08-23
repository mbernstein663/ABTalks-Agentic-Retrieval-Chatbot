# ABTalks AI Cohort — RAG Coverage Chatbot

## Project Description

This repository archives a 30-day applied AI engineering project built during the ABTalks Learning Cohort. The final system is a healthcare coverage assistant combining structured SQLite data, unstructured policy text, ChromaDB retrieval, local LLMs, LangChain/LangGraph agents, MCP tools, a FastAPI/Streamlit application, governance and evaluation, and container/Kubernetes deployment.

The insurance data is synthetic and the project is intended for learning and demonstration.

## #ABTalksOnAI

The ABTalks Learning Cohort is a free, hands-on program focused on building AI RAG applications from fundamentals through deployment. Each phase added a new layer: environment setup, data ingestion, retrieval, LLM behavior, application development, agents, governance, evaluation, and production infrastructure.

This repository preserves the code, notebooks, experiment outputs, documentation, and deployment artifacts produced throughout the program.

## Usage Instructions

1. Install dependencies with `pip install -r requirements.txt`.
2. Configure required environment variables in a local `.env`; do not commit secrets.
3. Ensure the Ollama model used by the project, `qwen2.5-coder:14b`, is available.
4. Run the full demo with `docker compose up --build`.
5. Open the Streamlit frontend at `http://localhost:8501`.
6. Backend health is available at `http://localhost:8000/health`.
7. Kubernetes manifests are stored in `k8s/`, with deployment observations in `k8s_notes.md`.

## Phase 1: Environment and Tooling

**Days 1–3 | July 2026**

Established the Python environment, dependency workflow, local Ollama experiments, API/UI practice, and Git/GitHub project structure. Early work is archived under `first-week-materials/`, with dependencies in `requirements.txt` and repository controls in `.gitignore`.

## Phase 2: Data Foundations

**Days 4–6 | July 20–21, 2026**

Built the structured and unstructured data foundation using `data/plans.csv`, `data/claims.csv`, `data/coverage.db`, `notebooks/day04-ingestion-lab.ipynb`, `day05-readingprocessingunstruct.ipynb`, and `day06-chunkingdocuments.ipynb`.

OCR and document extraction outputs were stored in `raw_text/` and transformed into chunked records in `knowledge_base.jsonl`.

## Phase 3: Embeddings and Vector Search

**Days 7–10 | July 22–23, 2026**

Created embeddings, inspected semantic relationships, populated ChromaDB, and tested retrieval quality.

Main artifacts: `notebooks/day07-vectorembeddings.ipynb`, `day08-ChromaDBsetup.ipynb`, `day09-populate-vectordb.ipynb`, `day10-advancedvectorsearch.ipynb`, `embeddings.npy`, `embeddings_2d.png`, `knowledge_base_embed.jsonl`, `chroma_db/`, and `retrieval_engine.py`.

## Phase 4: LLM Core, Prompting, and Fine Tuning

**Days 11–15 | July 30–August 1, 2026**

Connected retrieval to the LLM, refined RAG prompting and tool behavior, prepared training data, and tested LoRA/PEFT fine-tuning.

Key files: `rag_chatbot.py`, `tool_calling_chatbot.py`, `fine_tune_dataset.jsonl`, `fine_tune_train.jsonl`, `fine_tune_test.jsonl`, `json_check+split.py`, `fine_tuning_peft.py`, `fine_tuning_response_load.py`, and `qwen-lora/`.

## Phase 5: Chatbot Application Build

**Days 16–20 | August 3–6, 2026**

Converted the RAG pipeline into an application with a FastAPI backend, Streamlit frontend, streaming responses, evidence cards, caching, and conversation memory/compression.

Main files: `coverage-chatbot-api/main.py`, `app.py`, `response_cards.py`, `cache_utils.py`, and `memory_test.md`.

## Phase 6: Agentic AI & MCP

**Days 21–24 | August 7–10, 2026**

Added LangChain tool orchestration, specialist agents, routing, and an MCP server for structured claim and plan tools with fallbacks.

Core artifacts: `langchain_agent.py`, `multi_agent.py`, `multi_agent_comparison.md`, `mcp_server.py`, `mcp_test_notes.md`, `tool_call_log.md`, and `agent_traces.md`.

## Phase 7: Governance and Evaluation

**Days 25–27 | August 11–13, 2026**

Added input/output guardrails, PII redaction, rate and token governance, A/B testing, adversarial testing, and RAGAS evaluation.

Important files: `guardrails_config.py`, `redact_pii.py`, `token_utils.py`, `ab_test.py`, `ab_test_results.md`, `ragas_run.py`, `ragas_eval_set.jsonl`, `ragas_run_results.jsonl`, `ragas_scorecard.md`, `GOVERNANCE.md`, `adversarial_tests.md`, and `experiment_design.md`.

## Phase 8: Containerization, Kubernetes, and Production

**Days 28–30 | August 13–17, 2026**

Containerized the frontend and backend, composed the application with Ollama, deployed through Minikube/Kubernetes, added health probes and scaling/rollout tests, and explored production observability with Langfuse.

Main artifacts: `Dockerfile`, `Dockerfile.frontend`, `docker-compose.yml`, `docker_notes.md`, `k8s/backend-deployment.yaml`, `k8s/backend-service.yaml`, `k8s/frontend-deployment.yaml`, `k8s/frontend-service.yaml`, `k8s_notes.md`, `observability_notes.md`, and `chaos_test.md`.

---

**Archive status:** Cohort build completed and preserved as an end-to-end record of the learning process, implementation decisions, experiments, and deployment work.
