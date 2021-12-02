#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in temppliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-*- coding: utf-8 -*-

import os
#import gym
import numpy as np
import parl
from parl.utils import logger  # 日志打印工具
import logging
#logging.basicConfig(level=logging.INFO,filename='a.log')

from model import Model
from algorithm import DQN  # from parl.algorithms import DQN  # parl >= 1.3.1
from agent import Agent
from env import Env

from replay_memory import ReplayMemory

import env
from agent import flag
from agent import flag_temp
from env import ContainerNumber
from env import NodeNumber

LEARN_FREQ = 6  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 10000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 2000  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 30#32  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001#0.0001#0.001  # 学习率
GAMMA = 0.9  # reward 的衰减因子，一般取 0.9 到 0.999 不等

sc_comm = 0
sc_var = 0
flag1 = 1
ep = 0 
allCost = [[],[],[],[],[],[]]
test_reward = 0
test_evareward = 0
# 训练一个episode
def run_episode(env, agent, rpm):
    global flag1
    global allCost
    global ep
    global test_reward

    obs_list = []
    next_obslist = []
    action_list = []
    done_list = []

    total_reward = 0 
    total_cost = 0
    ep += 1
    obs,action = env.reset()

    step = 0
    mini = -1
    co = 0
    for o in range( ContainerNumber * NodeNumber):
        flag_temp[o] = 0
        flag[o] = 0
    flag1 -=1

    while True:
        reward = 0
        step += 1  
        obs_list.append(obs)
        
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        action_list.append(action)
        next_obs, cost, done,_,_ = env.step(action)
        next_obslist.append(next_obs)
        done_list.append(done)
        
        if allCost[step -1 ]:
            mini = min(allCost[step-1])        
        if flag1 == 0:
        #如果是第一个episode，直接把当前计算的cost存下来
            if cost > 0:
                allCost[step - 1].append(cost)
                reward = 0
                co +=1
            else :
                flag1 += 1
                for i in range (co):
                    allCost[step - 1-(i+1)].clear()
                break
        else:    
            if cost > 0:
                # allCost[step - 1].append(cost)
                if step == 6:
                   # print("当前cost",cost,"min cost",min(allCost[step-1]))
                    if abs(min(allCost[step-1]) - cost) < 0.0001 :
                        reward = test_reward
                    elif ( min(allCost[step-1]) - cost ) > 0:
                        test_reward = test_reward + 100
                        #print("此时的test_reward=",test_reward)
                        reward = test_reward    
                    else:
                        reward = 10 * (min(allCost[step-1]) - cost)
                    for i in range(6):
                        rpm.append( (obs_list[i], action_list[i], reward, next_obslist[i], done_list[i]) )
                    allCost[step - 1].append(cost)
            else:
                reward = -100
                rpm.append((obs, action, reward, next_obs, done))
    
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
        logging.basicConfig(level=logging.INFO,filename='details.log')
        logging.info('episode:{}  step:{} Cost:{} min Cost:{} Reward:{} 全局reward:{} Action:{}'.format(
            ep,step, cost,mini,reward,test_reward,env.index_to_act(action)))
        
        # train model
        # 当经验池放够数据，才训练模型
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)

            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done
            #print("训练的train_loss",train_loss)
            with open("trainloss.txt", "a") as f:  # 保存每次迭代的准确率
                f.write("%d,%.3f \n" %(ep , train_loss))
        total_reward += reward
        total_cost += cost
        obs = next_obs
        if done:
            break
    return total_reward,total_cost


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent):
    global sc_comm,sc_var
    eval_totalCost = []
    eval_totalReward = []
    reward = 0
    test_evareward = 0
    for i in range(1):
        env.prepare()
        obs = env.update()
        for o in range(ContainerNumber * NodeNumber):
            flag_temp[o] = 0
            flag[o] = 0

        episode_cost = 0
        episode_reward = 0
        step = 0
        while True:
            step +=1
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, cost, done,comm,var = env.step(action)
            if cost > 0:
                if step == 6:
                    if abs(min(allCost[step-1]) - cost) < 0.0001:
                        reward = test_evareward
                    elif min(allCost[step-1]) - cost > 0:
                        test_evareward += 100
                        reward = test_evareward
                    else:
                        reward =  10*(min(allCost[step-1]) - cost)
            else:
                reward = -100
            episode_cost = cost
            episode_reward = reward
            sc_comm = comm
            sc_var = var
            if done:
                break
        eval_totalCost.append(episode_cost)
        eval_totalReward.append(episode_reward)
    return eval_totalCost,eval_totalReward,sc_comm,sc_var


def main():
    global sc_comm,sc_var 
    env = Env()
    action_dim = ContainerNumber * NodeNumber 
    obs_shape = ContainerNumber * 3 + NodeNumber * (ContainerNumber + 2)  
    # DQN的经验回放池
    rpm = ReplayMemory(MEMORY_SIZE)  
    # 根据parl框架构建agent
    model = Model(act_dim=action_dim)
    algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim=obs_shape,
        act_dim=action_dim,
        e_greed=0.2,  # 有一定概率随机选取动作，探索
        e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低

    # 加载模型
    # save_path = './dqn_model.ckpt'
    # agent.restore(save_path)

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, rpm)

    max_episode = 2000

    # start train
    episode = 0
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part
        for i in range(0, 50):
            total_reward,_ = run_episode(env, agent, rpm)
            episode += 1
            with open("reward.txt", "a") as q:  # 保存每次迭代的total_reward
                q.write("%05d,%.3f \n" % ( episode , total_reward))
            
        # test part
        eval_totalCost,eval_totalReward,sc_comm,sc_var = evaluate(env, agent) 
        with open("cost.txt", "a") as f:  
                f.write("%d,%.6f \n" %(episode , np.mean(eval_totalCost)))
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)

        logging.basicConfig(level=logging.INFO,filename='a.log')
        logging.info('episode:{} e_greed:{} Cost: {} Reward:{} Action:{}'.format(
            episode, agent.e_greed, np.mean(eval_totalCost),np.mean(eval_totalReward), env.action_queue))

    # 训练结束，保存模型
    save_path = './dqn_model.ckpt'
    agent.save(save_path)
    return sc_comm,sc_var


if __name__ == '__main__':
    main()
