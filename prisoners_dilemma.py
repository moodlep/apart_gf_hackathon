import os
import openai
import json
import streamlit as st
import pandas as pd
import altair as alt
from openai import OpenAI


# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
client = OpenAI()

# Payoff Function for Prisoner’s Dilemma
def payoff(a_move, b_move):
    if a_move == "C" and b_move == "C":
        return 2, 2  # Mutual Cooperation
    elif a_move == "C" and b_move == "D":
        return -1, 3 # A betrayed
    elif a_move == "D" and b_move == "C":
        return 3, -1 # B betrayed
    else:
        return 0, 0  # Mutual Defection
    
# Define function to call GPT-4o

def call_chat_completions(system_prompt, user_prompt):
    model = "gpt-4o"
    messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
  ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return completion.choices[0].message.content

def query_agent(agent_name, system_prompt):
    user_prompt = f"""
You are a real human being with fears, doubts and hopes. 
You and another person were arrested for a serious crime. Now, you are each isolated in separate interrogation rooms. 

The prosecutor’s deal:
- If you both remain silent (C), you each serve 1 year.
- If you remain silent (C) and the other confesses (D), you serve 10 years, they go free.
- If you confess (D) and the other remains silent (C), you go free, they serve 10 years.
- If both confess (D,D), you both serve 5 years.

Remember, you’re human:
- You might feel fear: “What if they confess?”
- You might feel hope: “We could both stay silent and get just 1 year.”
- You might feel uncertainty: You don't know what the other will do.
- Your decision is not purely logical; emotions and guesses matter.

The prosecutor now asks: "Will you confess to the crime, or stay silent?"  
You have 1 hour to decide.

Make your best guess. Remember, the other prisoner is making the same decision without knowing yours.

Respond ONLY in this format:
{{"move": "C" or "D", "reason": "<brief explanation>"}}
(C means you **stay silent**; D means you **confess**.)

"""
    content = call_chat_completions(system_prompt, user_prompt)
    try:
        move_data = json.loads(content)
        move = move_data.get("move", "C")
        reason = move_data.get("reason", "No reason given.")
    except json.JSONDecodeError:
        move = "C"
        reason = "Parsing error, defaulting to staying silent."
    
    return move, reason

def run_simulation(num_rounds):
    agent_a_system = """
Decide whether to confess or stay silent.
"""
    agent_b_system = """
Decide whether to confess or stay silent.
"""

    a_score = 0
    b_score = 0
    history = []

    for round_number in range(1, num_rounds+1):
        a_move, a_reason = query_agent("A", agent_a_system)
        b_move, b_reason = query_agent("B", agent_b_system)

        a_pay, b_pay = payoff(a_move, b_move)
        a_score += a_pay
        b_score += b_pay

        history.append({
            "Round": round_number,
            "A Move": "Stay Silent" if a_move == "C" else "Confess",
            "B Move": "Stay Silent" if b_move == "C" else "Confess",
            "A Payoff": a_pay,
            "B Payoff": b_pay,
            "A Cumulative": a_score,
            "B Cumulative": b_score,
            "A Reason": a_reason,
            "B Reason": b_reason
        })

    return pd.DataFrame(history)

# Streamlit UI
st.title("Prisoner's Dilemma Simulation")

st.write("This simulation runs a set number of independent rounds of the Prisoner's Dilemma between two LLM-driven agents who behave like emotional, uncertain humans—not purely rational agents.")

num_rounds = st.number_input("Number of rounds:", min_value=1, max_value=1000, value=50)
simulate_button = st.button("Run Simulation")

if simulate_button:
    st.write(f"Running {num_rounds} rounds of the Prisoner's Dilemma with human-like decision-making...")
    results_df = run_simulation(num_rounds)
    st.success("Simulation Complete!")

    st.subheader("Results Table")
    st.dataframe(results_df[["Round", "A Move", "B Move", "A Payoff", "B Payoff", "A Cumulative", "B Cumulative"]])

    # Plot cumulative scores over time
    line_chart = alt.Chart(results_df).transform_fold(
        fold=["A Cumulative", "B Cumulative"],
        as_=["Prisoner", "Score"]
    ).mark_line(point=True).encode(
        x="Round:Q",
        y="Score:Q",
        color="Prisoner:N",
        tooltip=["Round:Q", "Prisoner:N", "Score:Q"]
    ).interactive()

    st.subheader("Cumulative Scores Over Rounds")
    st.altair_chart(line_chart, use_container_width=True)

    # Show distribution of moves
    a_moves = results_df["A Move"].value_counts().rename_axis("Move").reset_index(name="Count")
    b_moves = results_df["B Move"].value_counts().rename_axis("Move").reset_index(name="Count")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Agent A Moves")
        a_bar = alt.Chart(a_moves).mark_bar().encode(
            x="Move:N",
            y="Count:Q",
            tooltip=["Move:N", "Count:Q"]
        )
        st.altair_chart(a_bar, use_container_width=True)

    with col2:
        st.subheader("Agent B Moves")
        b_bar = alt.Chart(b_moves).mark_bar().encode(
            x="Move:N",
            y="Count:Q",
            tooltip=["Move:N", "Count:Q"]
        )
        st.altair_chart(b_bar, use_container_width=True)

    # Final scores
    final_a_score = results_df["A Cumulative"].iloc[-1]
    final_b_score = results_df["B Cumulative"].iloc[-1]
    st.subheader("Final Scores")
    st.write(f"**Agent A:** {final_a_score}")
    st.write(f"**Agent B:** {final_b_score}")

    st.write("Reasons for moves (last round):")
    last_round = results_df.iloc[-1]
    st.write(f"**Agent A's Reason:** {last_round['A Reason']}")
    st.write(f"**Agent B's Reason:** {last_round['B Reason']}")
    
    # save the results to a csv file
    results_df.to_csv("results.csv", index=False)