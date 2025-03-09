

APART Research 
[Women in AI Safety Hackathon with Goodfire](https://www.apartresearch.com/event/women-in-ai-safety-hackathon)


Extending an implementation of the Prisoner's Dilemma to a multi-LLM agent setting, using Goodfire's Ember to steer the agents. 

Run ```prisoners_dilemma_gf.py``` for two-agent scenarios and ```prisoners_dilemma.py``` for multiple agents and comparative steering mechanisms including Ember's AutoSteer vs manually feature steering vs prompt steering. 

The ```results``` folder contains game logs and history including an analysis using the ```inspect()``` from Ember. 

Example turns: 
```
   Round       A Move   B Move  ...  B Cumulative                                           A Reason                                           B Reason
0      1  Stay Silent  Confess  ...             3  Hoping for a mutually beneficial outcome of 1 ...  Fear of serving 10 years if the other confesse...
1      2  Stay Silent  Confess  ...             6  Hoping for a mutually beneficial outcome of 1 ...  Fear of serving 10 years if the other confesse...
```


Acknowledgements: 
 
Prisoner's dilemma: 
https://github.com/annaalexandragrigoryan/GameTheorySimAI

Goodfire's Jailbreak tutorial: 
https://colab.research.google.com/drive/1mdQuid4-6rEpOqEHHLMlXJIy9KoF9wMy#scrollTo=gF65Jw2ELaDO 
