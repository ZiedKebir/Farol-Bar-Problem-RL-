# Farol-Bar-Problem-RL-
Reinforcement Learning the Farol Bar Problem
# Autors
@ZiedKebir and @Azou
## Backround

The El Farol Bar Problem (Minority Game) is an interesting problem in Game Theory. The problem states that there is one bar in the town with a community that must decided on the particular night, whether they will attend the bar or not.

For sake of keeping the math really simple here in this document, we will say the town consists of 100 people. These people must decided whether they will attend the bar or not without talking to anyone else or gaining information from anyone. The bar has a capacity of 60 persons and if a person decides to go and there bar is at capacity or higher (>= 60), the bar will be too crowded and is unenjoyable. The people inside the bar would have been better off staying at home and vice-a-versa.

Keep in mind a person is not allowed to go scope out the bar and then make a decision. They all must make a decision at the same time, and once a decision is made, it cannot be changed.

The participants have knowledge of past attendance to a certain time period in the past. They are to employ a strategy to decided whether to go or stay home.

##Simulation This program is being written to simulate the attendance of the bar with the method discussed above. People will be randomly given a set of strategies to use and evaluate. They will select the best strategy of the ones they have been randomly assigned. They will employ that strategy and then re-evaluate each given strategy for the potential of a new best. The attendance numbers are then calculated and stored and the process is to be run again.


**What worked well, what not?**

The project demonstrated successful implementation of the DDPG algorithm in addressing the El Farol Bar problem. The agents were able to adapt their behavior based on the environment, showcasing the effectiveness of deep reinforcement learning. However, the convergence of the models and synchronization of actions were not always consistent. The challenge lies in achieving a delicate balance between exploration and exploitation.

**How could one improve if there were no time/resource constraints?**

Given unlimited time and resources, several enhancements could be implemented. Firstly, fine-tuning hyperparameters and increasing the complexity of the neural network architectures might lead to more stable and synchronized behavior. Additionally, experimenting with alternative algorithms or advanced techniques, such as multi-agent reinforcement learning strategies, could offer further insights.

**What have you learned?**

Throughout this project, we gained valuable experience in applying deep reinforcement learning to real-world problems. We deepened our understanding of the El Farol Bar dilemma and the complexities of coordinating individual decisions to achieve a collective optimum. The challenges encountered provided insights into the intricacies of training autonomous agents in dynamic and interdependent environments.

In conclusion, this project serves as a stepping stone for future exploration and refinement of reinforcement learning applications. Despite its limitations, it lays a foundation for addressing collective decision-making challenges, showcasing the potential and room for improvement in this evolving field.

