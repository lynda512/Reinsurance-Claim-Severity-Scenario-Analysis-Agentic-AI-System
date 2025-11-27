
# src/agent/agent.py
import os
import json
from . import tools

# Optionally use OpenAI; keep usage optional and pluggable
USE_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
if USE_OPENAI:
    import openai

def simple_llm(prompt):
    """
    Very small deterministic LLM-like summarizer for offline demos.
    Use OpenAI if API key exists.
    """
    if USE_OPENAI:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini", # example; change to available
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=300
        )
        return resp['choices'][0]['message']['content']
    # offline fallback
    return f"MOCK-LLM SUMMARY FOR PROMPT:\n{prompt[:800]}...\n\n(Use OPENAI_API_KEY to enable real LLM responses.)"

def run_agent(prompt):
    """
    Very small agent flow:
    - Parse intent keywords from prompt (heuristic)
    - Call appropriate tools
    - Compose a summary using LLM (or simple template)
    """
    intent = prompt.lower()
    out = {'calls': []}
    if 'data quality' in intent or 'quality' in intent:
        df = tools.load_df(limit=5)
        out['calls'].append({'tool': 'query_data', 'result_count': len(df)})
    if 'predict' in intent or 'severity' in intent or 'estimate' in intent:
        sample = tools.load_df(limit=100).head(20)
        recs = sample.to_dict(orient='records')
        preds = tools.run_model_on_records(recs)
        out['calls'].append({'tool': 'run_model', 'n': len(preds), 'sample_mean_pred': float(sum(preds)/len(preds))})
    if 'scenario' in intent or 'shock' in intent:
        sim = tools.simulate_scenario(shock_pct=0.2, subset_filters=None, sample_size=100)
        out['calls'].append({'tool':'simulate_scenario', 'result': sim})
    # build prompt for LLM
    llm_prompt = "You are a reinsurance analyst assistant. The user asked:\n\n" + prompt + "\n\nTool outputs:\n" + json.dumps(out, indent=2)
    summary = simple_llm(llm_prompt)
    return {'summary': summary, 'tool_outputs': out}
