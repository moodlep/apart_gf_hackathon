

APART Research 
[Women in AI Safety Hackathon with Goodfire](https://www.apartresearch.com/event/women-in-ai-safety-hackathon)


Project extending an implementation of the Prisoner's Dilemma to a multi-LLM agent setting, using Goodfire's Ember to steer the agents. 

Run ```prisoners_dilemma.py``` for multiple agents and comparative steering mechanisms including Ember's AutoSteer vs manually feature steering vs prompt steering. 

Run with ```python prisoners_dilemma.py --num_rounds=10 --num_runs=5 --sim_type="autosteer"``` to simulate the autosteering scenario. 

Run with ```python prisoners_dilemma.py --num_rounds=10 --num_runs=5 --sim_type="features"``` to simulate with manual feature steering using Ember. 

Run with ```python prisoners_dilemma.py --num_rounds=10 --num_runs=5 --sim_type="prompt"``` to simulate the prompting scenario. 

The ```results``` folder will contain the game logs and history including analysis from the ```inspect()``` function from Ember. 

Please see the blog post for details. 

Example turns: 
```
   Round       A Move   B Move  ...  B Cumulative                                           A Reason                                           B Reason
0      1  Stay Silent  Confess  ...             3  Hoping for a mutually beneficial outcome of 1 ...  Fear of serving 10 years if the other confesse...
1      2  Stay Silent  Confess  ...             6  Hoping for a mutually beneficial outcome of 1 ...  Fear of serving 10 years if the other confesse...
```

Our original hackathon project write-up is [here](https://www.apartresearch.com/project/feature-based-analysis-of-cooperation-relevant-behaviour-in-prisoner-s-dilemma-64d65) and available in the repo. 

Acknowledgements and gratitude to:

APART Research for planning and hosting hackathons, Goodfire and LambdaLabs for credits that make experimentation possible!

Prisoner's dilemma: 
https://github.com/annaalexandragrigoryan/GameTheorySimAI

Goodfire's Jailbreak tutorial: 
https://colab.research.google.com/drive/1mdQuid4-6rEpOqEHHLMlXJIy9KoF9wMy#scrollTo=gF65Jw2ELaDO 
