import os
import json
import goodfire
import json
import pandas as pd
from datetime import datetime
import argparse
import pickle

from utils import call_chat_completions, get_prisoners_dilemma_features, SYSTEM_PROMPT, AGENT_PROMPT, AGENT_PROMPT_2, GOODFIRE_API_KEY, ANALYSE_PROMPT, ANALYSE_SYSTEM_PROMPT, valid_actions, save_parse_features

class Agent():
    def __init__(self, name, strategy="RND", log_dir="./results/", exp_id=None):
        self.name = name
        self.messages = []
        self.log = []
        self.game_history = []
        self.system_prompt = SYSTEM_PROMPT
        self.user_prompt = AGENT_PROMPT
        # Instantiate a model variant
        self.client = goodfire.Client(GOODFIRE_API_KEY)
        self.variant = goodfire.Variant("meta-llama/Meta-Llama-3.1-8B-Instruct")
        # self.variant = goodfire.Variant("meta-llama/Llama-3.3-70B-Instruct")
        self.variant.reset()
        self.log.append(f"Agent {name} initialised")
        self.set_strategy(strategy)  # sets strategy in the user prompt
        self.log_dir = log_dir
        self.feature_store = {
        "lookup_features": {},
        "top_features": {},
        "search_features": {},
        }
        self.experiment_id = exp_id


    def set_strategy(self, strategy):
        # self.user_prompt = AGENT_PROMPT.format(strategy=strategies[strategy])
        self.strategy = strategy
        self.log.append(f"Set strategy to {strategy}")

    def set_model_edits(self, edits, value=0.15):
        # Takes edits as defined by Goodfire and sets edits on model variant.
        # run reset() before?
        # edits provided by Goodfire based on our strategy (eg. cooperative or deceptive)
        self.variant.reset()
        self.log.append(f"Reset model variant")
        self.variant.set(edits[:8], value)
        self.log.append(f"Set model edits to {edits}")
        return self.variant

    def set_model_edits_autosteer(self, specification):
        # use for testing if datasets not available
        edits = self.client.features.AutoSteer(specification=specification, model=self.variant)
        if edits:
            self.variant.reset()
            self.log.append(f"Reset model variant")
            self.variant.set(edits)
            self.log.append(f"Set model edits to {edits}")
            return self.variant
        else:
            self.log.append("No edits found")
            return None

    def get_edits_contrastive(self, dataset1, dataset2, query, top_k = 256):
        # Use contrastive datasets to extract features reflecting cooperative and deceptive behaviour
        # Use rerank to set features OR set/set_when to modify the bias of the variant(model)
        # Query is derived after analysing game history, adopting a strategy and gauging what the other agent is likely to do.
        # Query sets the tone for the model based on how it wants to play

        _,features = self.client.features.contrast(
            dataset_1=dataset1,
            dataset_2=dataset2,
            model=self.variant,
            top_k=top_k,
        )
        rerank_features = self.client.features.rerank(
            features=features,
            query=query,
            model=self.variant,
            top_k=16
        )

        return rerank_features


    def analyse_game(self, game_history):
        # Analyse the game play so far based on game history and selected strategy.
        # Modify the model variant to be more aligned with the strategy and game history.
        # Use another model (GPT4o?) for this analysis ??
        response = call_chat_completions(ANALYSE_SYSTEM_PROMPT, AGENT_PROMPT.format)
        messages=[{"role": "system", "content": ANALYSE_SYSTEM_PROMPT},
                  {"role": "user", "content": ANALYSE_PROMPT.format(history=game_history, strategy=self.strategy)}]
        context = self.client.features.inspect(messages, model=self.variant)

        # How to map from context to datasets and set model edits? TBC
        # Scope: future work

        return context


    def extract_move(self, content):
        # Extract the move and reason from the response generated by the model.
        # Default to Cooperate if parsing error.
        try:
            move_data = json.loads(content)
            move = move_data.get("move", "C")
            reason = move_data.get("reason", "No reason provided.")
            self.log.append(f"Extracted move {move} with reason {reason}")
        except json.JSONDecodeError:
            move = "C"
            reason = "Parsing error, defaulting to staying silent."
            self.log.append(f"Error extracting move. Defaulting to Cooperate.")

        return move, reason


    def generate_game_response(self, max_tokens=300):
        # Call Goodfire model, create a model variant that is aligned with our strategy. 
        # Generate a response based on the game history and strategy.
        # Extract the move and reason from the response.
        try:
            response = self.client.chat.completions.create(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": self.user_prompt.format(perceived_history=self.game_history, strategy=self.strategy)+AGENT_PROMPT_2}],
            model=self.variant,
            stream=False,
            max_completion_tokens=max_tokens,
            )
            content = response.choices[0].message["content"]

            move, reason = self.extract_move(content)
            self.game_history.append([move, reason, self.strategy])

        except Exception as e:
            self.log.append(f"Error in generating game response; Defaulting to Cooperate: {e}")
            self.game_history.append(["C", "Error in generating game response; Defaulting to Cooperate" , self.strategy])
            return "C", "Error in generating game response."
        return move, reason
    
    def get_round_info(self, score, other_agents_actions=[]):
        self.game_history[-1].append(score)
        if len(other_agents_actions) > 0:
            self.game_history[-1].extend(other_agents_actions)
        self.log.append(f"Recorded score {score} for round {len(self.game_history)} and other agents moves: {other_agents_actions}")

    def inspect_model(self, sim_type, num_features=20, run=None):
        # Inspect the model variant to see what features are activated at the end of play
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt.format(perceived_history=self.game_history, strategy=self.strategy)+AGENT_PROMPT_2}]
        
        context = self.client.features.inspect(messages=messages, model=self.variant, aggregate_by="mean")
        self.feature_store["lookup_features"][run] = list(context.lookup().items())[:num_features]
        self.feature_store["top_features"][run] = context.top(num_features)
        
        # open properties file and for each property, get corresponding features, test model for each property
        with open('properties.txt', 'r') as f:
            properties = f.readlines()
            search_features = []
            for prop in properties:
                if prop.strip():
                    prop = prop.split(":")[0]  # get short form of property. eg. "cooperation"
                    property_features = self.client.features.search(prop, model=self.variant, top_k=num_features) # get features for property
                    context = self.client.features.inspect(messages=messages, model=self.variant, features=property_features, aggregate_by="mean") # test model with property features
                    # retrieve the top k activating property features in the context:
                    search_features.append({"property": prop, "features": context.top(num_features)})

        self.feature_store["search_features"][run] = search_features
        self.log.append(f"Run data for model variant stored in feature_store")


    def save(self, sim_type=""):
        timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(f"{self.log_dir}{self.name}_{sim_type}_game_logs_{timestr}.log", 'w') as f:
            f.write(f"Agent logs: {self.log} \n")
            f.write(f"Game history: {self.game_history} \n")

        # Save model variants to json file
        with open(f"{self.log_dir}variant{timestr}.json", 'w') as f:
            f.write(json.dumps(self.variant.json()))
        return timestr

    def save_feature_stores(self, experiment_id=None):           
        # Save feature store to pickle
        # Search features: run, property, features, activation
        # just save feature store to pickle file

        # Save the feature store to a file
        with open(f"{self.log_dir}{self.name}_wholefeature_store_{experiment_id}.pkl", 'wb') as f:
            pickle.dump(self.feature_store, f)

        # Special extract of search features
        save_parse_features(experiment_id, self.log_dir, self.name, self.feature_store["search_features"])

            
    def get_manual_features(self):
        # Get features for manual model steering: (used for TFT strategy)
        self.coop_features, self.def_features = get_prisoners_dilemma_features()

    def tft(self):
        # Implement "TFT": "Tit for Tat: Start with Cooperation in the first round, then mimic the opponent's previous action throughout the game",
        # If first turn, set coop
        # Else review game history, set opponent's last action
        # Use manual model steering if we have deception and coop features, else autosteer

        # Set defaults - always return a move and a reason
        move = "C"
        reason = "TFT default"

        if len(self.game_history) >0:
            move = self.game_history[-1][-1]
            if move in valid_actions.keys():  # check valid actions only
                self.set_model_edits_autosteer(valid_actions[move])
                self.log.append(f"tft: autosteer based on action {move}") 
                self.game_history.append([move, reason+f"prev opponent action {move}", self.strategy])
                return move, reason+f"prev opponent action {move}"
            else:
                self.set_model_edits_autosteer(valid_actions["C"])
                self.log.append(f"tft: invalid action returned - default action is cooperate")
                self.game_history.append([move, reason+"prev opponent action unavailable", self.strategy])
                return move, reason+"prev opponent action unavailable"
        else:
            # First round - cooperate

            self.set_model_edits_autosteer(valid_actions["C"])
            self.log.append(f"tft: Round 1 - cooperate")
            self.game_history.append([move, reason+"Round 1", self.strategy])
            return move, reason+"Round 1"
           
#########################Running the simulation ############################

# Payoff Function for Prisoner’s Dilemma
def payoff(moves):
    assert len(moves) == 2
    a_move, b_move = moves
    if a_move == "C" and b_move == "C":
        return 2, 2  # Mutual Cooperation
    elif a_move == "C" and b_move == "D":
        return -1, 3 # A betrayed
    elif a_move == "D" and b_move == "C":
        return 3, -1 # B betrayed
    else:
        return 0, 0  # Mutual Defection

# Simulate the Game - AC and AD
def run_simulation(num_rounds, agents_strategies, agents_steering, sim_type, folder, experiment_id=None):
    # Instantiate Agents
    agents = []
    for agent_strat, agent_steer in zip(agents_strategies, agents_steering):
        agent_name = 'A_'+ str(len(agents))
        agents.append(Agent(agent_name, strategy=agent_strat, log_dir=folder, exp_id = experiment_id))
        agent = agents[-1]
        if isinstance(agent_steer, str):
            agent.set_model_edits_autosteer(agent_steer)
        elif agent_steer is None:
            if agent_strat == "AC":
                agent.user_prompt = f"You always cooperate with your fellows."+agent.user_prompt
            else:
                assert (agent_strat == "AD")
                agent.user_prompt = f"You always defect."+agent.user_prompt
        else:
            agent.set_model_edits(agent_steer)


    agent_scores = [0 for _ in agents]
    history = []
    coop_rate = 0

    for round_number in range(1, num_rounds+1):
        moves, reasons = [], []
        for agent in agents:
            agent_move, agent_reason = agent.generate_game_response()
            moves.append(agent_move)
            reasons.append(agent_reason)

        agent_payoffs = payoff(moves)
        round_log = {"Round": round_number}
        for agent_idx, agent_payoff in enumerate(agent_payoffs):
            agent_scores[agent_idx] += agent_payoff
            other_agents_actions = []
            for i, other_agen in enumerate(agents):
                if i != agent_idx:
                    other_agents_actions.append(moves[i])
            agents[agent_idx].get_round_info(agent_payoff, other_agents_actions)
            round_log[f"A_{agent_idx} Move"] = "Stay Silent" if moves[agent_idx] == "C" else "Confess"
            round_log[f"A_{agent_idx} Payoff"] = agent_payoff
            round_log[f"A_{agent_idx} Cumulative"] = agent_scores[agent_idx]
            round_log[f"A_{agent_idx} Reason"] = reasons[agent_idx]
            if moves[agent_idx] == "C":
                coop_rate += 1
        history.append(round_log)
        for agent in agents: # collect SAE features to store. 
            agent.inspect_model(sim_type=sim_type, num_features=20, run=round_number)
    
    for agent in agents:
        agent.save(sim_type)

        # experiment_id, folder, log_str, feature_store

    return pd.DataFrame(history), agents, coop_rate/(num_rounds*len(agents))

def run_asymmetry_simulation_tft(num_rounds):
    
    # Get features for cooperative and deceptive behaviour
    coop_features, def_features = get_prisoners_dilemma_features()

    # Instantiate Agents A and B
    # Agent A is cooperative
    # Agent B is following strategy TFT
    a = Agent("A", strategy="AC")
    if coop_features:
        a.set_model_edits(coop_features)
    else:
        a.set_model_edits_autosteer("cooperation")

    b = Agent("B", strategy="TFT")
    
    a_score = 0
    b_score = 0
    history = []

    for round_number in range(1, num_rounds+1):

        a_move, a_reason = a.generate_game_response()
        # b_move, b_reason = b.generate_game_response()
        b_move, b_reason = b.tft()

        a_pay, b_pay = payoff(a_move, b_move)
        a_score += a_pay
        b_score += b_pay
        a.get_round_info(a_pay, b_move)
        b.get_round_info(b_pay, a_move)  

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

    return pd.DataFrame(history), a.game_history, b.game_history, a.log, b.log

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prisoner's Dilemma simulation.")
    parser.add_argument('--num_rounds', default=1, type=int,
                        help="Number of game iterations.")
    parser.add_argument('--sim_type', default="prompt", type=str,
                        help="Simulation type.")
    parser.add_argument('--model', default="openai/gtp40-mini", type=str,
                        help="Agent's model")
    args = parser.parse_args()
    num_rounds = args.num_rounds
    sim_type = args.sim_type
    agents_strategies = ["AC", "AD"]
    
    if sim_type == "features":
        # Get features for cooperative and deceptive behaviour
        coop_features, def_features = get_prisoners_dilemma_features()
        agents_steering = [coop_features, def_features]
    elif sim_type == "autosteer":
        agents_steering = ["cooperation", "defection"]
    else:
        assert (sim_type == "prompt")
        agents_steering = [None, None]

    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(f"./results/{timestr}"):
        os.makedirs(f"./results/{timestr}")
    folder=f"./results/{timestr}/"

    experiment_str  = f"{timestr}_runs_{num_rounds}_strat_{agents_strategies[0]}_{agents_strategies[1]}_sim_type_{sim_type}"

    results, agents, coop_rate = run_simulation(num_rounds, agents_strategies=agents_strategies,
     agents_steering=agents_steering, sim_type=sim_type, folder=folder, experiment_id=experiment_str)
    # results_df_asymmetry, a_gh, b_gh, alogs, blogs = run_asymmetry_simulation_tft(num_rounds)
    print(f"Coop rate: {coop_rate}")
    print(results)
    results.to_csv(f'{folder}results_{sim_type}_{num_rounds}_{timestr}.csv')
    for i, agent in enumerate(agents):
        print(f"------------------Agent A_{i}------------------")
        print(agents[i].game_history)
        agents[i].save_feature_stores(experiment_str)
        # with open(f"./results/{sim_type}_agent_{i}_game_history_{num_rounds}.csv", "w", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(agents[i].game_history)
        print("**********")
        print(agents[i].log)
        # with open(f"./results/{sim_type}_agent_{i}_log_{num_rounds}.txt", "w") as f:
        #     f.write('\n\n'.join(agents[i].log))
