# !pip install streamlit torch transformers langgraph pydantic

import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing import List

# ─── STEP 1: Define Pydantic Models for the State Schema ────────────────

class Alternative(BaseModel):
    name: str
    values: List[float]  # Order: [is_safe_enough, reliab_center_m, skilled_and_trained, cost_optimized]

class StateModel(BaseModel):
    alternatives: List[Alternative]
    best_alternative: str = ""
    best_score: float = 0.0
    recommendation: str = ""

# ─── STEP 2: Define Node Functions for the LangGraph Workflow ───────────

def calculate_best_alternative(state: StateModel) -> StateModel:
    """
    Calculate the weighted sum for each alternative and select the best one.
    Weights:
      - is_safe_enough: 0.41
      - reliab_center_m: 0.26
      - skilled_and_trained: 0.19
      - cost_optimized: 0.14
    """
    weights = [0.41, 0.26, 0.19, 0.14]
    best_score = None
    best_alternative = ""
    for alt in state.alternatives:
        score = sum(v * w for v, w in zip(alt.values, weights))
        if best_score is None or score > best_score:
            best_score = score
            best_alternative = alt.name
    
    # Update the state and return it
    updated_state = state.copy(update={
        "best_alternative": best_alternative,
        "best_score": best_score
    })
    return updated_state

def generate_recommendation(state: StateModel) -> StateModel:
    """
    Generate a professional recommendation using a lightweight model.
    """
    prompt = (
        f"As a professional civil engineer with over 10 years of project management experience, "
        f"I recommend adopting {state.best_alternative} for its optimal balance of safety, reliability, "
        f"skill requirements, and cost efficiency."
    )
    # Use the lightweight model "google/flan-t5-small"
    MODEL_NAME = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    generated_text = generator(prompt, max_length=100)[0]['generated_text']

    # Update the state with the generated recommendation and return the state
    updated_state = state.copy(update={"recommendation": generated_text})
    return updated_state

# ─── STEP 3: Build the LangGraph Workflow ─────────────────────────────────────

# Use the Pydantic model as the state schema.
state_schema = StateModel

graph = StateGraph(state_schema=state_schema)
graph.add_node("calculate", calculate_best_alternative)
graph.add_node("recommend", generate_recommendation)
graph.set_entry_point("calculate")
graph.add_edge("calculate", "recommend")
graph.return_key = "recommend"  # Set the return key

# Compile the graph to obtain a runnable object
runnable_graph = graph.compile()

# ─── STEP 4: Build the Streamlit User Interface ──────────────────────────────

st.title("Decision Generator with LangGraph")
st.write("Enter the alternatives and their criteria values to receive a professional recommendation.")

# Define the criteria in the desired order
criteria = ["is_safe_enough", "reliab_center_m", "skilled_and_trained", "cost_optimized"]

# Let the user select the number of alternatives
num_alternatives = st.number_input("Enter the number of alternatives", min_value=1, value=1, step=1)

# Create a form for entering alternative details
alternatives = []
with st.form("alternatives_form"):
    for i in range(int(num_alternatives)):
        st.subheader(f"Alternative {i+1}")
        alt_name = st.text_input(f"Enter name for alternative {i+1}", key=f"name_{i}")
        values = []
        for criterion in criteria:
            val = st.number_input(
                f"{criterion} (0-1)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                key=f"{criterion}_{i}"
            )
            values.append(val)
        if alt_name:
            alternatives.append(Alternative(name=alt_name, values=values))
    submitted = st.form_submit_button("Make Decision")

# ─── STEP 5: Process Input & Display Results ────────────────────────────────

if submitted:
    if not alternatives:
        st.error("Please provide at least one alternative with a valid name.")
    else:
        # Prepare the state using the Pydantic model
        state = StateModel(alternatives=alternatives)
        # Call invoke on the runnable_graph object
        result_state = runnable_graph.invoke(state)
        
        # Since result_state is a dict-like AddableValuesDict, access using keys
        st.subheader("Decision Result")
        st.markdown(f"**Selected Alternative:** {result_state['best_alternative']}")
        st.markdown(f"**Weighted Score:** {result_state['best_score']}")
        st.markdown("**Professional Recommendation:**")
        st.write(result_state['recommendation'])
