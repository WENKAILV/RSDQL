#-*- coding: utf-8 -*-
import xml.dom.minidom
from xml.etree import ElementTree as ET

CPUnum = 7
Mem = 8*1024

dom1 = xml.dom.minidom.parse('D:\\RL code\\hotel\\dataSet\\data.xml')
root = dom1.documentElement
dom2 = ET.parse('D:\\RL code\\hotel\\dataSet\\data.xml')

class Data():
    def __init__(self):
        self.service_containernum = []
        self.service_container = []
        self.service_container_relationship = []
        self.container_state_queue = []
        self.NodeNumber = int(root.getElementsByTagName('nodeNumber')[0].firstChild.data)
        self.ContainerNumber = int(root.getElementsByTagName('containerNumber')[0].firstChild.data)
        self.ServiceNumber = int(root.getElementsByTagName('serviceNumber')[0].firstChild.data)
        self.ResourceType = int(root.getElementsByTagName('resourceType')[0].firstChild.data)

        for oneper in dom2.findall('./number/containerNumber'):
            for child in oneper:
                self.service_container_relationship.append(int(child.text))

        for oneper in dom2.findall('./number/serviceNumber'):
            for child in oneper:
                self.service_containernum.append(int(child.text))
                self.service_container.append([int(child[0].text)])
                self.container_state_queue.append(-1)
                self.container_state_queue.append(int(child[0][0].text)/CPUnum)
                self.container_state_queue.append(int(child[0][1].text)/Mem)

        Dist_temp = []
        for oneper in dom2.findall('./distance'):
            for child in oneper:
                Dist_temp.append((float(child.text)))
        self.Dist = [ Dist_temp[ i : i + self.NodeNumber] for i in range(0,len(Dist_temp),self.NodeNumber)]

        weight_temp = []
        for oneper in dom2.findall('./weight'):
            for child in oneper:
                weight_temp.append((float(child.text)))
        self.service_weight = [ weight_temp[ i : i + self.ServiceNumber] for i in range(0,len(weight_temp),self.ServiceNumber)]







