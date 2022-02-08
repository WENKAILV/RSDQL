import pandas as pd
import numpy as np
from env import Env
from env import ContainerNumber, NodeNumber, ServiceNumber, service_containernum, service_container
import train
sigma = 0.9
delta = 0.3
alpha = 0.5
beta = 0.5

df = pd.read_excel('D:\\RL code\\dqn_final\\monitor.xlsx',header=0) 
df2 = pd.read_excel('D:\\RL code\\dqn_final\\cost.xlsx',header=0) 
data = df.values[1]
data2 = df2.values[0]
usage = [0] * ServiceNumber
Avg = [0] * ServiceNumber

def SCM():
    k = -1
    decision = -1
    for i in range(ServiceNumber):
        for j in range(service_containernum[i]):
            usage[i] += alpha * data[j+i+1] + beta * data[j+i+1+ContainerNumber]
        Avg[i] = usage[i] / service_containernum[i]
        print("Avg[",i,"]",Avg[i])
        if Avg[i] > sigma:
            k = i
            decision = 1
        elif Avg[i] < delta:
            k = i
            decision = 0
    return k,decision

def elastic(k,decision):
    container = service_container[k][0]
    Cost = data2[0]
    Var = data2[1]
    if decision == 1:
        score = float('inf')
        for i in range(NodeNumber):
            env = Env()
            for j in range(ContainerNumber):
                action = int(data2[j+2])
                env.step(action)
            env.step(i*ContainerNumber+container)
            _,tCost, tVar = env.cost()
            score_cost = beta * (tCost - Cost)
            score_var = alpha * (tVar - Var)     
            print("on node",i,",score is ",score_var + score_cost )
            if score > (score_var + score_cost):
                index = i
                score = score_var + score_cost
    elif decision == 0:
        score = float('-inf')
        for i in range( NodeNumber-1, -1, -1):
            env = Env()
            for j in range(ContainerNumber):
                for k in range(ContainerNumber):
                    if j != k:
                        action = int(data2[k+2])
                        env.step(action)
            _,tCost, tVar = env.cost()
            score_cost = beta * (tCost - Cost)
            score_var = alpha * (tVar - Var)
            if score < (score_var + score_cost):
                index = i
                score = score_var + score_cost
    return index,container

k,decision = SCM()
print("scaling container is",k,", and decision is ",decision)
index,container = elastic(k,decision)
print("elastice scaling is to deploy container",container,"on node", index)
