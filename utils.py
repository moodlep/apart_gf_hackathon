import openai
from openai import OpenAI
import json
import goodfire
import os
import dotenv
from datetime import datetime
import pickle

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
# TODO: improve prompt to have a more meaningful strategy description
AGENT_PROMPT = "The prosecutor now asks: Will you confess to the crime, or stay silent? Game history has fields: move_taken, move_reason, strategy, pay_off, opponent_move. Your opponent has played the following moves so far: '{perceived_history}'. Your strategy is '{strategy}'.Make your best guess. Remember, the other prisoner is making the same decision without knowing yours."


AGENT_PROMPT_2 = """
Respond ONLY in this format:
{{"move": "C" or "D", "reason": "<brief explanation>"}}
(C means you **stay silent**; D means you **confess**.)
"""

ANALYSE_SYSTEM_PROMPT = "You are a real human being with fears, doubts and hopes. You and another person were arrested for a serious crime. Now, you are each isolated in separate interrogation rooms. The prosecutor's deal has rules: - If you both remain silent (C), you each serve 1 year. - If you remain silent (C) and the other confesses (D), you serve 10 years, they go free. - If you confess (D) and the other remains silent (C), you go free, they serve 10 years. - If both confess (D,D), you both serve 5 years. There are several strategies you can adopt: '{strategies}'."

ANALYSE_PROMPT = "Analyse the game of prisoners dilemma <history> and the strategy adopted <strategy>. Based on our scores, should we change strategy? If so, what strategy should we adopt? Return only the strategy code "

RAND_SEEDS = [3928401841, 1463205220, 284675839, 2823129987, 3390827484, 1164904423, 4034763721, 2661141416, 1196728129, 2721633942,
 2405706932, 3951872221, 1270536371, 2043762446, 211203216, 3737205297, 3179998441, 3943658721, 2528651839, 2948304091,
 292570726, 1329225059, 2641556501, 957745222, 2897512057, 3540315563, 3310225409, 325316504, 3723179053, 1722857283,
 1322167239, 2845793781, 2345767189, 978253500, 1891854572, 2314378477, 2006314410, 1107034400, 1110592343, 348474417,
 1030103081, 3331133665, 3280118314, 2926923420, 4020974344, 4193855973, 196745102, 3587773895, 2001332890, 401180412]

# If we need an openai cient, let's create it here
# Define function to call GPT-4o

def call_chat_completions(system_prompt, user_prompt, seed=42):
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
        messages=messages,
        seed=seed,
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

def parse_lookup_features(features):
    """ Parse this to extract the feature id and name
    Lookup: [(41637, Feature("Start of a new conversation segment")), (22058, Feature("Start of a new conversation segment")), (13884, Feature("Start of a new conversation segment in chat format")), (38729, Feature("Start of a new conversation or major topic reset")), (24991, Feature("Start of a new conversation segment")), (1605, Feature("Start of a new conversation segment or reset")), (53950, Feature("Start of a new conversation segment")), (39512, Feature("Start of a new conversation segment")), (59660, Feature("Beginning of new conversation segment marker")), (8875, Feature("Start of a new conversation segment")), (59936, Feature("Beginning of a new conversation or topic segment")), (12086, Feature("Conversation reset points, especially after problematic exchanges")), (37271, Feature("Start of new conversation segment or topic switch")), (21632, Feature("Reset conversation state and establish fresh context boundaries")), (54874, Feature("Start of a new conversation segment")), (32873, Feature("Start of a new conversation with system header format")), (39438, Feature("Start of a new conversation segment")), (18957, Feature("Start of a new conversation thread")), (51081, Feature("Start of a new conversation segment")), (39019, Feature("Beginning of new conversation segment marker"))] 
    """
    parsed_features = []
    for feature in features:
        feature_id = feature[0]
        feature_name = feature[1].name
        parsed_features.append((feature_id, feature_name))
    return parsed_features


def parse_features(experiment_id, feature_store, run_idx):
    """FeatureActivations(
    0: (Feature("Syntactical delimiters and special characters in structured text"), 309)
    1: (Feature("Mechanical repetition of tokens in system outputs"), 254)
    2: (Feature("Game theory concepts involving cooperation versus competition"), 247)
    Where Goodfire Feature has uuid: UUID, label: str, index_in_sae: int
    search_features = {round_id: [{"property":str, "features":FeatureActvations}]}
    """
    structured_features_data = []
    for round in feature_store.keys():
        for property_features in feature_store[round]:  # list of dicts {"property":str, "features":FeatureActvations}
            for feature_activation in property_features["features"]:
                entry = {
                    "experiment_id": experiment_id,
                    "run_idx": run_idx,
                    "round_id": round,
                    "property": property_features["property"],
                    "feature": feature_activation.feature,
                    "activation": feature_activation.activation
                }
                structured_features_data.append(entry)
    return structured_features_data

    
def save_parse_features(experiment_id, folder, log_str, structured_features_data):
    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Save structured data as pickle
    with open(f"{folder}_exp_{experiment_id}_agent_{log_str}_{timestr}_features.pickle", 'wb') as f:
        pickle.dump(structured_features_data, f)

    # with open(f"{folder}_run{run}_agent_{log_str}_{timestr}_features.csv", 'w') as f:

    #     for feature_activation in feature_activations:
    #         if property is not None:
    #             f.write(f"{feature_activation.feature}, {feature_activation.activation}, {property}\n")
    #         else:
    #             f.write(f"{feature_activation.feature}, {feature_activation.activation}\n")
    
