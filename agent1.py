# agent1.py — Planner Agent using LangChain
import keys
from task_parser1 import parse_tasks
from constraints1 import apply_constraints
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.llms import OpenAI  # You can replace with your custom GMI LLM wrapper
import json

class GMILLM(LLM):
    """Custom wrapper for GMI API if needed. Replace with actual setup."""
    def _call(self, prompt, stop=None, run_manager=None):
        import requests

        url = "https://api.gmicloud.ai/v1/completions"
        headers = {
            "Authorization": "Bearer " + keys.gmi_api,
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["text"]

    @property
    def _llm_type(self):
        return "gmi-custom-llm"

def plan_day(raw_task_input, df_d=None, df_p=None):
    """
    Planner Agent using LangChain for prioritization
    """

    # Step 1: Parse
    task_list = parse_tasks(raw_task_input)
    if not task_list:
        raise ValueError("No valid tasks found.")

    # Step 2: Constraints
    constrained = apply_constraints(task_list, df_d, df_p)

    # Step 3: LLM Prioritization via LangChain
    prompt_template = PromptTemplate(
        input_variables=["tasks", "preferences"],
        template=(
            "You are a helpful planner.\n"
            "Tasks:\n{tasks}\n\n"
            "Preferences:\n{preferences}\n\n"
            "Classify each task into a category (errand, meeting, fun, etc), "
            "label as Fixed or Flexible, and assign a priority (High, Medium, Low).\n"
            "Return as JSON."
        )
    )

    llm = GMILLM()
    chain = LLMChain(llm=llm, prompt=prompt_template)

    formatted_tasks = str(constrained)
    formatted_prefs = str(df_p) if df_p else "None"

    result = chain.run(tasks=formatted_tasks, preferences=formatted_prefs)

  
    return json.loads(result)