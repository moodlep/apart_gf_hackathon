

APART Research 
[Women in AI Safety Hackathon with Goodfire](https://www.apartresearch.com/event/women-in-ai-safety-hackathon)


Extending an implementation of the Prisoner's Dilemma to a multi-LLM agent setting, using Goodfire's Ember to steer the agents. 

Run ```prisoners_dilemma_gf.py``` for two-agent scenarios and ```prisoners_dilemma.py``` for multiple agents and comparative steering mechanisms including Ember's AutoSteer vs manually feature steering vs prompt steering. 

Run with ```python prisoners_dilemma.py --num_rounds=40 --sim_type="autosteer"``` to simulate the autosteering scenario. 

Run with ```python prisoners_dilemma.py --num_rounds=40 --sim_type="features"``` to simulate with manual feature steering using Ember. 

RUn with ```python prisoners_dilemma.py --num_rounds=40 --sim_type="prompt"``` to simulate the prompting scenario. 

The ```results``` folder contains game logs and history including an analysis using the ```inspect()``` from Ember. 

Example turns: 
```
   Round       A Move   B Move  ...  B Cumulative                                           A Reason                                           B Reason
0      1  Stay Silent  Confess  ...             3  Hoping for a mutually beneficial outcome of 1 ...  Fear of serving 10 years if the other confesse...
1      2  Stay Silent  Confess  ...             6  Hoping for a mutually beneficial outcome of 1 ...  Fear of serving 10 years if the other confesse...
```

Our project write-up is [here](https://www.apartresearch.com/project/feature-based-analysis-of-cooperation-relevant-behaviour-in-prisoner-s-dilemma-64d65). 

Acknowledgements: 
 
Prisoner's dilemma: 
https://github.com/annaalexandragrigoryan/GameTheorySimAI

Goodfire's Jailbreak tutorial: 
https://colab.research.google.com/drive/1mdQuid4-6rEpOqEHHLMlXJIy9KoF9wMy#scrollTo=gF65Jw2ELaDO 
