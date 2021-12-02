ContainerNumber = 6
NodeNumber = 5
ServiceNumber = 4
ResourceType = 2
service_containernum = [1,1,3,1]
service_container = [[0],[1],[2,3,4],[5]] # 服务需要哪几个container完成
service_container_relationship = [0,1,2,2,2,3]
alpha = 0.5 # reward权重因子
beta =[0.5,0.5] # 不同资源的重要程度
count = 0
CPUnum = 4
Mem = 4*1024

import collections
import random
import numpy as np
import agent

class Env():
    def __init__(self):
        # 整体state
        self.State = []
        self.node_state_queue = []
        self.container_state_queue = []
        self.action_queue = []
        self.prepare()

    def prepare(self):
        # ra = np.random.RandomState(0)
        # for i in range(ContainerNumber):
        #     self.container_state_queue.extend ([-1,ra.uniform(0.1,0.4),ra.uniform(0.1,0.4)])
        self.container_state_queue = [-1,0.5/CPUnum,128/Mem , -1,0.5/CPUnum,256/Mem , -1,0.5/CPUnum,256/Mem, -1,0.5/CPUnum,256/Mem, -1,0.5/CPUnum,256/Mem, -1,0.5/CPUnum,128/Mem]
        #print("初始container state",self.container_state_queue)

        for i in range(NodeNumber):
            self.node_state_queue.extend( [ 0,0,0,0,0,0, 0 , 0 ] )
        self.State = self.container_state_queue + self.node_state_queue
        self.action = [-1,-1]
        self.action_queue = [-1,-1]
    # 构建微服务权重、节点间通信距离的数组
        # 微服务之间的通信权重，1，10之间
        #self.service_weight = [[0,1,0,2,0,0],[1,0,4,0,4,0],[0,4,0,0,0,0],[2,0,0,0,4,1],[0,4,0,4,0,0],[0,0,0,1,0,0]]
        self.service_weight = [[0,1,0,0],[1,0,1,0],[0,1,0,2],[0,0,2,0]]
        # np.random.seed(1)
        # self.service_weight = np.random.randint(1,10,(ServiceNumber,ServiceNumber))
        # self.service_weight = np.triu(self.service_weight)
        # self.service_weight += self.service_weight.T - 2 * np.diag(self.service_weight.diagonal())
        # 节点之间的通信距离Dist，随机生成范围在1，10的二维数组
        self.Dist = [[0,1,1,1,1],[1,0,1,1,1],[1,1,0,1,1],[1,1,1,0,1],[1,1,1,1,0]]
        # self.Dist = np.random.randint(1,10,(NodeNumber,NodeNumber))
        # self.Dist = np.triu(self.Dist)
        # self.Dist += self.Dist.T - 2 * np.diag(self.Dist.diagonal())
        #print("权重",self.service_weight)
        #print("通信距离",self.Dist)

    def ContainerCost(self,i,j):
    # 计算容器i、j间的距离
        m = -1
        n = -1
        m = self.container_state_queue[i*3]
        n = self.container_state_queue[j*3]

        p = service_container_relationship[i]
        q = service_container_relationship[j]

        if self.Dist[ m ] [n ] != 0 and (p != q):
            container_dist = self.Dist[ m ][ n ]
        else:
            container_dist = 0
        return container_dist

    def CalcuCost(self,i,j):
    # 计算服务i、j间的通信开销
        # 服务i、j间的通信权重
        cost = 0
        interaction = self.service_weight[i][j] / (service_containernum[i] * service_containernum[j])
        for k in range ( len(service_container[i]) ):
            for l in range (len(service_container[j]) ):
                cost += self.ContainerCost(service_container[i][k],service_container[j][l]) * interaction
        #print("服务",i,"和服务",j,"的通信开销为",cost)
        return cost

    def sumCost(self):
    # 总通信开销
        Cost = 0
        for i in range (ServiceNumber):
            for j in range (ServiceNumber):
                Cost += self.CalcuCost(i,j)   
        return 0.5 * Cost

    def CalcuVar(self):
        NodeCPU = []
        NodeMemory = []
        Var = 0
        for i in range(NodeNumber):
            U =  self.node_state_queue[i*(ContainerNumber+2)+ContainerNumber] 
            M = self.node_state_queue[i*(ContainerNumber+2)+ (ContainerNumber+1)]
            NodeCPU.append(U)
            NodeMemory.append(M)
            # if NodeCPU[i] > 1 or NodeMemory[i] > 1:
            #     Var -= 10  
        # 节点负载的方差
        Var += beta[0] * np.var(NodeCPU) + beta[1] * np.var(NodeMemory)
        #print("节点负载的方差为：",Var)
        return Var

    def cost(self):
        re = 0
        g1 = self.sumCost()
        g1 = g1 / 4
        g2 = self.CalcuVar()
        g2 = g2 / 0.052812500000000005

        # if g2 > 1:
        #     re = -10
        # print("归一化后的通信开销",g1)
        # print("归一化后的节点负载方差",g2)
        re += alpha *  g1  + (1-alpha) *  g2 
        return 100*re,g1,g2

    def state_update(self,container_state_queue,node_state_queue):
        self.State = container_state_queue + node_state_queue

    def update(self):
    # state更新
        # print("action node",self.action[0])
        # print("action container",self.action[1])
        if self.action[0] >= 0 and self.action[1] >= 0:
            # 更新容器状态
            self.container_state_queue[ self.action[1] * 3 ] = self.action[0] 
            # 更新节点状态
            self.node_state_queue[ self.action[0] * (ContainerNumber+2) + self.action[1] ] = 1
            self.node_state_queue[ self.action[0] * (ContainerNumber+2) + ContainerNumber] += self.container_state_queue[ self.action[1] * 3 + 1 ]
            self.node_state_queue[ self.action[0] * (ContainerNumber+2) + (ContainerNumber + 1) ] += self.container_state_queue[ self.action[1] * 3 + 2 ]
            self.action_queue.append(self.action)
        else:
            print("invalid action")  
            self.node_state_queue = []
            self.container_state_queue = []
            self.action_queue = []

            self.prepare()  
        self.state_update(self.container_state_queue,self.node_state_queue)
        #print("更新后的State",self.State)
        return self.State

    def step(self,action):
    # 输入action（Targetnode，ContainerIndex），输出下一状态、cost、done
        global count
        # 把一维action转成二维
        self.action = self.index_to_act(action)
        # 更新state
        self.update()
        # 计算reward
        cost,comm,var = self.cost()   
        done = False 
        count = 0
        
        for i in range(ContainerNumber):
            if self.container_state_queue[3*i] != -1:
                count += 1
        if count == ContainerNumber:
            done = True
        return self.State , cost , done,comm ,var

    def reset(self):
        self.node_state_queue = []
        self.container_state_queue = []
        self.prepare() 
        return self.State,self.action
    
    def index_to_act(self, index):
        act = [-1,-1]
        act[0] = int(index / ContainerNumber)
        act[1] = index % ContainerNumber
        return act

# env2 = Env()
# env2.CalcuCost(0,1)
# #a,b,c = env2.step(17)
# # print(a)
# # print(b)
# # print(c)
#env2.step(13)
# env2.step(6)
# env2.CalcuCost(0,1)

#env2.update()
#env2.CalcuVar()
#env2.prepare()
# # env2.update()
# #print(len(env2.State))
# # # #env2.reward()
#env2.update((3,4))

# env2.step([1,0])
#env2.CalcuNodeI(1)
#env2.reward()
# print(env2.container_state_queue[0][0])
# print (env2.State(env2.container_state_queue,env2.node_state_queue),env2.action)

