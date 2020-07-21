# AI-Robo-Advisor
### Autonomous Robo-Advisor based on Genetic Algorithm and Evolutionary Strategy using Deep Neural Networks.
* The Autonomous Robo-Advisor is endowed with partial consciousness and the ability to understand in real-time his own level of quality and his perspectives.

* He realizes how well he succeeds in achieving his goal in the initially given data space.

* If it becomes aware that it is not able to achieve the goal in the current data space, it decides to expand the information search space for better knowledge.

* Then it evolves, restarting the online self-learning process, taking as the primary initialization of genes (weights) for a set number of parallel workers, the genes of the best workers from the previous population.

* He seeks to predict as accurately as possible not only the future course change but also the time of this change in a given time-space.

* Its main goal is the quality and profitable management of the financial portfolio, absolutely without human intervention.

* The brain of the model architecture consists of two neural networks consisting of LSTM and CNN layers, which are simultaneously trained online in one session, to comprehend the limited, changing financial world and time dependencies.

* Since trading on a cryptocurrency exchange can be considered as a partially observable Markov decision-making process (POMDP), due to the fact that we do not fully see the state of the environment, we do not know how many other agents are in the environment and do not know the essence of their trading strategies, then for In the continuous process of exploring the huge strategic space of the cryptocurrency exchange, a genetic algorithm and an evolutionary strategy with an unlimited number of parallel workers were implemented.
