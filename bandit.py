#!/usr/bin/python

# Importing necessary libraries
import sys
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt

instance = sys.argv[2]
algorithm = sys.argv[4]
randomSeed = int(sys.argv[6])
epsilon = float(sys.argv[8])
horizon = int(sys.argv[10])
p = np.loadtxt(instance, delimiter=',')
n_arms = len(p)
hint_ts = np.sort(p)
max_reward = horizon*(max(p))

# Function to generate random 0-1 rewards corresponding to an arm a
def generate_rewards(arm):
    global p
    if (random.uniform(0, 1.0) < p[arm]):
        return 1
    else:
        return 0

def epsilon_greedy(epsilon, n_arms, horizon, max_reward,randomSeed):
    random.seed(randomSeed)
    rewards = [0 for _ in range(n_arms)]
    pulls_count = [0 for _ in range(n_arms)]
    emp_mean = [0 for _ in range(n_arms)]
    a = 0
    for _ in range(horizon):
        x = random.uniform(0,1.0)
        if (x < epsilon):
            a = random.randint(0,n_arms-1)
        else:
            a = np.argmax(emp_mean)
        r = generate_rewards(a)
        rewards[a] += r
        pulls_count[a] += 1
        emp_mean[a] = (rewards[a]/pulls_count[a])
    # print(emp_mean)
    return (max_reward - sum(rewards))

def UCB(horizon, max_reward, n_arms,randomSeed):
    random.seed(randomSeed)
    ucb_val = [0 for _ in range(n_arms)]
    rewards = [0 for _ in range(n_arms)]
    emp_mean = [0 for _ in range(n_arms)]
    pulls_count = [0 for _ in range(n_arms)]
    for t in range(min(n_arms,horizon)):
        r = generate_rewards(t)
        rewards[t] = r
        pulls_count[t] = 1
        emp_mean[t] = r
    for t in range(n_arms,horizon):
        for i in range(n_arms):
            ucb_val[i] = emp_mean[i] + math.sqrt((2*(math.log(t)))/pulls_count[i])
        a = np.argmax(ucb_val)
        r = generate_rewards(a)
        rewards[a] += r
        pulls_count[a] += 1
        emp_mean[a] = (rewards[a]/pulls_count[a])
    # print(emp_mean)
    return (max_reward - sum(rewards))

def KL_UCB(horizon, max_reward, n_arms, c, randomSeed):
    random.seed(randomSeed)
    kl_ucb_val = [0 for _ in range(n_arms)]
    rewards = [0 for _ in range(n_arms)]
    emp_mean = [0 for _ in range(n_arms)]
    pulls_count = [0 for _ in range(n_arms)]
    for t in range(min(n_arms,horizon)):
        r = generate_rewards(t)
        rewards[t] = r
        pulls_count[t] = 1
        emp_mean[t] = r
    for t in range(n_arms,horizon):
        val = math.log(t)+(c*(math.log(math.log(t))))
        precision = 1e-6
        for i in range(n_arms):
            l = emp_mean[i]
            r = 1
            while (r-l) > precision:
                mid = (l+r)/2
                if emp_mean[i] == 0:
                    kl = ((1-emp_mean[i])*(math.log((1-emp_mean[i])/(1-mid))))
                elif emp_mean[i] == 1:
                    kl = ((emp_mean[i])*(math.log(emp_mean[i]/mid)))
                else:                        
                    kl = ((emp_mean[i])*(math.log(emp_mean[i]/mid)))+((1-emp_mean[i])*(math.log((1-emp_mean[i])/(1-mid))))
                if (pulls_count[i]*kl) <= val:
                    l = mid
                else:
                    r = mid
            kl_ucb_val[i] = l
        a = np.argmax(kl_ucb_val)
        r = generate_rewards(a)
        rewards[a] += r
        pulls_count[a] += 1
        emp_mean[a] = (rewards[a]/pulls_count[a])
    # print(emp_mean)
    return (max_reward - sum(rewards))

def thompson_sampling(horizon, max_reward, n_arms, randomSeed):
    random.seed(randomSeed)
    s = [0 for _ in range(n_arms)]
    rewards = [0 for _ in range(n_arms)]
    emp_mean = [0 for _ in range(n_arms)]
    pulls_count = [0 for _ in range(n_arms)]
    for _ in range(horizon):
        x = []
        for i in range(n_arms):
            x.append(random.betavariate(s[i]+1,pulls_count[i]-s[i]+1))
        a = np.argmax(x)
        r = generate_rewards(a)
        rewards[a] += r
        if r == 1:
            s[a] += 1
        pulls_count[a] += 1
        emp_mean[a] = (rewards[a]/pulls_count[a])
    # print(emp_mean)
    return (max_reward - sum(rewards))

def thompson_sampling_with_hint(horizon, max_reward, n_arms, hint_ts, randomSeed):
    random.seed(randomSeed)
    belief = [[1/(n_arms) for _ in range(n_arms)] for _ in range(n_arms)]
    rewards = [0 for _ in range(n_arms)]
    emp_mean = [0 for _ in range(n_arms)]
    pulls_count = [0 for _ in range(n_arms)]
    for t in range(horizon):
        x = []
        for i in range(n_arms):
            x.append(belief[i][-1])
        a = np.argmax(x)
        r = generate_rewards(a)
        if r == 1:
            for i in range(n_arms):
                belief[a][i] = (belief[a][i]*hint_ts[i])/(np.dot(belief[a],hint_ts))
        else:
            for i in range(n_arms):
                belief[a][i] = (belief[a][i]*(1-hint_ts[i]))/(np.dot(belief[a],1-hint_ts))
        rewards[a] += r
        pulls_count[a] += 1
        emp_mean[a] = (rewards[a]/pulls_count[a])
    # print(hint_ts)
    # print(emp_mean)
    # print(pulls_count)
    return (max_reward - sum(rewards))

if algorithm == 'epsilon-greedy':
    print("{0}, {1}, {2}, {3}, {4}, {5}".format(instance,algorithm,randomSeed,epsilon,horizon,epsilon_greedy(epsilon, n_arms, horizon, max_reward, randomSeed)))
elif algorithm == 'ucb':
    print("{0}, {1}, {2}, {3}, {4}, {5}".format(instance,algorithm,randomSeed,epsilon,horizon,UCB(horizon, max_reward, n_arms, randomSeed)))
elif algorithm == 'kl-ucb':
    print("{0}, {1}, {2}, {3}, {4}, {5}".format(instance,algorithm,randomSeed,epsilon,horizon,KL_UCB(horizon, max_reward, n_arms,3, randomSeed)))
elif algorithm == 'thompson-sampling-with-hint':
    print("{0}, {1}, {2}, {3}, {4}, {5}".format(instance,algorithm,randomSeed,epsilon,horizon,thompson_sampling_with_hint(horizon, max_reward, n_arms, hint_ts, randomSeed)))
else:
    print("{0}, {1}, {2}, {3}, {4}, {5}".format(instance,algorithm,randomSeed,epsilon,horizon,thompson_sampling(horizon, max_reward, n_arms, randomSeed)))


# Code to generate data in the outputDataT1.txt file
# e = [[0.4,0.8],[0.4,0.3,0.5,0.2,0.1],[0.15,0.23,0.37,0.44,0.50,0.32,0.78,0.21,0.82,0.56,0.34,0.56,0.84,0.76,0.43,0.65,0.73,0.92,0.10,0.89,0.48,0.96,0.60,0.54,0.49]]
# epsilon = 0.02
# num = 0
# for i in ('../instances/i-1.txt','../instances/i-2.txt','../instances/i-3.txt'):
#     instance = i
#     p = e[num]
#     n_arms = len(p)
#     hint_ts = np.sort(p)
#     for algo in ('epsilon-greedy','ucb','kl-ucb','thompson-sampling'):
#         algorithm = algo
#         for t in (100,400,1600,6400,25600,102400):
#             horizon = t
#             max_reward = horizon*(max(p))
#             for seed in range(50):
#                 randomSeed = seed
#                 random.seed(seed)
#                 if algorithm == 'epsilon-greedy':
#                     print("{0}, {1}, {2}, {3}, {4}, {5}".format(instance,algorithm,randomSeed,epsilon,horizon,epsilon_greedy(epsilon, n_arms, horizon, max_reward)))
#                 elif algorithm == 'ucb':
#                     print("{0}, {1}, {2}, {3}, {4}, {5}".format(instance,algorithm,randomSeed,epsilon,horizon,UCB(horizon, max_reward, n_arms)))
#                 elif algorithm == 'kl-ucb':
#                     print("{0}, {1}, {2}, {3}, {4}, {5}".format(instance,algorithm,randomSeed,epsilon,horizon,KL_UCB(horizon, max_reward, n_arms,3)))
#                 else:
#                     print("{0}, {1}, {2}, {3}, {4}, {5}".format(instance,algorithm,randomSeed,epsilon,horizon,thompson_sampling(horizon, max_reward, n_arms, randomSeed)))
#     num += 1


# Code to generate data in the outputDataT2.txt file
# e = [[0.4,0.8],[0.4,0.3,0.5,0.2,0.1],[0.15,0.23,0.37,0.44,0.50,0.32,0.78,0.21,0.82,0.56,0.34,0.56,0.84,0.76,0.43,0.65,0.73,0.92,0.10,0.89,0.48,0.96,0.60,0.54,0.49]]
# epsilon = 0.02
# num = 0
# for i in ('../instances/i-1.txt','../instances/i-2.txt','../instances/i-3.txt'):
#     instance = i
#     p = e[num]
#     n_arms = len(p)
#     hint_ts = np.sort(p)
#     for algo in ('thompson-sampling','thompson-sampling-with-hint'):
#         algorithm = algo
#         for t in (100,400,1600,6400,25600,102400):
#             horizon = t
#             max_reward = horizon*(max(p))
#             for seed in range(50):
#                 randomSeed = seed
#                 if algorithm == 'thompson-sampling':
#                     print("{0}, {1}, {2}, {3}, {4}, {5}".format(instance,algorithm,randomSeed,epsilon,horizon,thompson_sampling(horizon, max_reward, n_arms, randomSeed)))
#                 else:
#                     print("{0}, {1}, {2}, {3}, {4}, {5}".format(instance,algorithm,randomSeed,epsilon,horizon,thompson_sampling_with_hint(horizon, max_reward, n_arms, hint_ts, randomSeed)))
#     num += 1