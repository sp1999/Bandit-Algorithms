# Bandit-Algorithms

Implemented Epsilon-greedy, UCB, KL-UCB, Thompson sampling and Thompson sampling with hint bandit algorithms

The bandit.py file contains the implementation of each algorithm.

Run the command "python bandit.py --instance in --algorithm al --randomSeed rs --epsilon ep --horizon hz" where in is a path to the instance file, al is one of epsilon-greedy, ucb, kl-ucb, thompson-sampling, and thompson-sampling-with-hint, rs is a non-negative integer, ep is a number in [0,1] and hz is a non-negative integer to run the bandit algorithm on the bandit instance given and return a reward.

bandit.py outputs a single line with six entries, separated by commas and terminated with a newline ('\n') character which is given in outputFormat.txt

outputDataT1.txt contains the output data for every combination of
**instance** from "../instances/i-1.txt"; "../instances/i-2.txt"; "../instances/i-3.txt",
**algorithm** from epsilon-greedy with epsilon set to 0.02; ucb, kl-ucb, thompson-sampling,
**horizon** from 100; 400; 1600; 6400; 25600; 102400, and
**random seed** from 0; 1; ...; 49.

outputDataT2.txt contains the output data for every combination of
**instance** from "../instances/i-1.txt"; "../instances/i-2.txt"; "../instances/i-3.txt",
**algorithm** from thompson-sampling, thompson-sampling-with-hint,
**horizon** from 100; 400; 1600; 6400; 25600; 102400, and
**random seed** from 0; 1; ...; 49.

The bandit instances are located in the instances directory.

Run the check.sh script to see that all the command line parameters are read correctly and to print the output.

Run the command "python verifyOutput.py" to check that the output is printed in the correct format and is reproducible.

The generating_data.ipynb notebook contains the code for generating the data in outputDataT1.txt and outputDataT2.txt for comparing the performance of all the bandit algorithms and plotting their graphs.

## Results

![1_T1](https://github.com/sp1999/Bandit-Algorithms/blob/main/Results/1_T1.png) 
![1_T2](https://github.com/sp1999/Bandit-Algorithms/blob/main/Results/1_T2.png)

![1_T1](https://github.com/sp1999/Bandit-Algorithms/blob/main/Results/2_T1.png) 
![1_T2](https://github.com/sp1999/Bandit-Algorithms/blob/main/Results/2_T2.png)

![1_T1](https://github.com/sp1999/Bandit-Algorithms/blob/main/Results/3_T1.png) 
![1_T2](https://github.com/sp1999/Bandit-Algorithms/blob/main/Results/3_T2.png)

All the assumptions, observations, results and explanation of the algorithms is given in report.pdf

## Reference
https://www.cse.iitb.ac.in/~shivaram/teaching/cs747-a2020/pa-1/programming-assignment-1.html
