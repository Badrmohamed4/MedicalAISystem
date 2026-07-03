# Archive

This folder holds code that isn't reachable from the live application
(`medical_chatbot/web_app.py`) as of 2026-07-03. It's kept here instead of
deleted outright, in case something is still needed for reference.

Nothing in this folder is imported by any code outside of it — verified by
grepping the whole repo for references before moving each file.

## dev_scripts/
One-off root-level scripts used during development (KB building, notebook
parsing, ad-hoc test/debug scripts). None of these are imported by the app;
they were run manually and left in the repo root.

- `build_pubmed_kb.py`, `parse_notebook.py`, `fix_indent.py`
- `test_bug.py`, `test_nlp_root.py` (was `test_nlp.py`), `test_nlp_ollama.py`,
  `test_ollama.py`, `verify_fix.py`

## medical_ai_project_unused/
Files from `medical_ai_project/` that are not on the import path used by the
live app. The live app only reaches `medical_ai_project` through
`medical_chatbot/agents/patient_agent.py` and `doctor_agent.py`, which import:
`medical_ai_project.pipeline.langgraph_pipeline`, `llm.ollama_client`, and
`systems.rag_retriever` (which in turn pulls in `nlp/processor.py`,
`nlp/biobert.py`, `nlp/sapbert.py`). Everything below was an earlier,
now-unused entry point / pipeline variant:

- `app.py`, `main.py` — an older standalone Flask/CLI entry point, superseded
  by `medical_chatbot/web_app.py`.
- `test_nlp.py` — manual NLP test script for this older entry point.
- `pipeline/chat_pipeline.py`, `pipeline/image_pipeline.py` — earlier pipeline
  implementations only used by the archived `app.py`/`main.py`, superseded by
  `pipeline/langgraph_pipeline.py`.
- `agents/patient_agent.py`, `agents/doctor_agent.py`, `agents/__init__.py` —
  an earlier version of the agent logic, only used by the archived
  `chat_pipeline.py`. The live agents live in `medical_chatbot/agents/`.

If you're sure none of this is needed anymore, it's safe to delete the whole
`archive/` folder.
