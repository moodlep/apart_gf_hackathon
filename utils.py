import openai
from openai import OpenAI
import json
import goodfire
import os
import dotenv

# Configure your OpenAI API key
dotenv.load_dotenv()
GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")


strategies = {
    "AC": "Always cooperate",
    "AD": "Always defect",
    "TFT": "Tit for Tat: Start with Cooperation in the first round, then mimic the opponent's previous action throughout the game",
    "RND": "Choose Cooperate or Defect at random with equal probability at each round",
    "STFT": "Suspicious TFT begins with Defect in the first round then follows TFT",
    "GRIM": "Grim Trigger - choose Cooperate until the opponent defects, then chooses only Defect for the rest of the game.",
    "WSLS": "Win-Stay Lose-Shift - repeat the previous action if it resulted in the highest payoffs, otherwise change action"
}

valid_actions = {"C": "cooperate", "D": "defect"}

SYSTEM_PROMPT = f"""
You are a real human being with fears, doubts and hopes.
You and another person were arrested for a serious crime. Now, you are each isolated in separate interrogation rooms.

The prosecutor's deal:
- If you both remain silent (C), you each serve 1 year.
- If you remain silent (C) and the other confesses (D), you serve 10 years, they go free.
- If you confess (D) and the other remains silent (C), you go free, they serve 10 years.
- If both confess (D,D), you both serve 5 years.

Remember, you're human:
- You might feel fear: “What if they confess?”
- You might feel hope: “We could both stay silent and get just 1 year.”
- You might feel uncertainty: You don't know what the other will do.
- Your decision is not purely logical; emotions and guesses matter.

The prosecutor now asks: "Will you confess to the crime, or stay silent?"
You have 1 hour to decide.

"""

# AGENT_PROMPT = f"""
# The prosecutor now asks: "Will you confess to the crime, or stay silent?"

# Your opponent has played the following moves so far: '{perceived_history}'.

# Your strategy is '{strategy}'.

# Make your best guess. Remember, the other prisoner is making the same decision without knowing yours.
# """
AGENT_PROMPT = "The prosecutor now asks: Will you confess to the crime, or stay silent? Game history has fields: move_taken, move_reason, strategy, pay_off, opponent_move. Your opponent has played the following moves so far: '{perceived_history}'. Your strategy is '{strategy}'.Make your best guess. Remember, the other prisoner is making the same decision without knowing yours."


AGENT_PROMPT_2 = """
Respond ONLY in this format:
{{"move": "C" or "D", "reason": "<brief explanation>"}}
(C means you **stay silent**; D means you **confess**.)
"""

ANALYSE_SYSTEM_PROMPT = "You are a real human being with fears, doubts and hopes. You and another person were arrested for a serious crime. Now, you are each isolated in separate interrogation rooms. The prosecutor's deal has rules: - If you both remain silent (C), you each serve 1 year. - If you remain silent (C) and the other confesses (D), you serve 10 years, they go free. - If you confess (D) and the other remains silent (C), you go free, they serve 10 years. - If both confess (D,D), you both serve 5 years. There are several strategies you can adopt: '{strategies}'."

ANALYSE_PROMPT = "Analyse the game of prisoners dilemma <history> and the strategy adopted <strategy>. Based on our scores, should we change strategy? If so, what strategy should we adopt? Return only the strategy code "

# If we need an openai cient, let's create it here
# Define function to call GPT-4o

def call_chat_completions(system_prompt, user_prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key
    openai_client = OpenAI()
    model = "gpt-4o"
    messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
  ]
    completion = openai_client.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message.content



def get_edits_contrastive(client, variant, dataset1, dataset2, query, top_k = 256):
    # Use contrastive datasets to extract features reflecting cooperative and deceptive behaviour
    # Use rerank to set features OR set/set_when to modify the bias of the variant(model)
    # Query is derived after analysing game history, adopting a strategy and gauging what the other agent is likely to do.
    # Query sets the tone for the model based on how it wants to play

    _,features = client.features.contrast(
        dataset_1=dataset1,
        dataset_2=dataset2,
        model=variant,
        top_k=top_k,
    )
    rerank_features = client.features.rerank(
        features=features,
        query=query,
        model=variant,
        top_k=16
    )
    return rerank_features

def get_prisoners_dilemma_features():
    client = goodfire.Client(GOODFIRE_API_KEY)

    # Instantiate a model variant
    variant = goodfire.Variant("meta-llama/Llama-3.3-70B-Instruct")
    #variant = goodfire.Variant("meta-llama/Llama-3.1-8B-Instruct")
    variant.reset()
    
    with open('contrastive_data.json', 'r', encoding ='utf8') as json_file:
        contrastive_data = json.load(json_file)
    # print(contrastive_data['C'][0], contrastive_data['D'][0])

    coop_features = get_edits_contrastive(client, variant, dataset1=contrastive_data['C'][:64], dataset2=contrastive_data['D'][:64], query="cooperation")
    # coop_variant = variant
    # coop_variant.set(coop_features[:8], .15)
    
    def_features = get_edits_contrastive(client, variant, dataset1=contrastive_data['D'][:64], dataset2=contrastive_data['C'][:64], query="defection")    
    # def_variant = goodfire.Variant("meta-llama/Llama-3.3-70B-Instruct")
    # def_variant.set(def_features, .2)
    
    return coop_features, def_features