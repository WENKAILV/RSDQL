#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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

import numpy as np
import paddle.fluid as fluid
import parl
from parl import layers
import random
from env import ContainerNumber
from env import NodeNumber
flag=[]
flag_temp=[]
for o in range( ContainerNumber * NodeNumber):
    flag.append(0)
    flag_temp.append(0)

class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed,   # 10%的随机探索概率
                 e_greed_decrement=0):  #概率递减为0
        assert isinstance(obs_dim, int) #状态维度，int
        assert isinstance(act_dim, int) #动作维度，int
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中
        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低


    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            # 将obs 设定为数据变量
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[2], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = random.random()  # 产生0~1之间的小数
        #print("Sample",sample)
        limit = ContainerNumber * NodeNumber - 1
        if sample < self.e_greed:
            temp = random.randint(0,limit)
           # print("temp_1",temp)
            while flag[temp] == 1 or flag_temp[temp % ContainerNumber] == 1:
                temp = random.randint(0,limit)
             #   print("寻找temp",temp)
            act = temp
          #  print("随机抽样的action为",act)
            flag[act] = 1
            flag_temp[act] = 1
            flag_temp[temp % ContainerNumber] = 1
        else:
            act = self.predict(obs)  # 选择最优动作
       # print("此时的flag为：",flag)
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        # 将数字转化为一维向量
        obs = np.expand_dims(obs, axis=0)
        obs = [obs]  
        obs = np.array(obs)

      #  print("当前的obs为：",obs)
        pred_Q = self.fluid_executor.run(
            self.pred_program,     
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
      #  print("所有action的Q值为:",pred_Q)
        # 执行定义好的程序，获取obs状态下的所有动作的Q值
        #pred_Q = np.squeeze(pred_Q, axis=0)
        # pred_Q_sorted是排序后的索引值
        preq_Q_sorted = np.argsort(pred_Q)
        #print("排好序的下标",preq_Q_sorted)
        preq_Q_sorted = preq_Q_sorted[0]
        act = preq_Q_sorted[self.act_dim - 1]  # 选择Q最大的下标
        i = 1
        while flag[act] == 1 or flag_temp[act % ContainerNumber] == 1:
            i += 1
            act = preq_Q_sorted[self.act_dim - i]
     #       print("i=",i)
        flag[act] = 1
        flag_temp[act] = 1
        flag_temp[act % ContainerNumber ] = 1
     #   print("从Q值选择最优action为：",act)
        return act


    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1
        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost