import random as rng
import numpy as np
import matplotlib.pyplot as plt
from draw_mql import *

# Arm, leg, arm leg. Remember to flip the left from the right (left = 9 - 
# right value)
# draw_mql(9, 15, 0, 1, 1, 9, 9, 4, 3, 6, 7)


sex_cdf = {
    (2.0/3.0): 'male',
    1.0: 'female'
}

# 0 is hexagon (female) and 1 is circle (male)
bt_obs_m_cdf = {
    0.3: 0,
    1.0: 1
}

bt_obs_f_cdf = {
    0.7: 0,
    1.0: 1
}

# Body type is basically determined 1-1 from sex, so just do a mapping here
# 1 is circle, 0 is hex
sex_bt_map = {
    'male': 1,
    'female': 0
}

bt_obs_bt_map = {
    0: bt_obs_f_cdf,
    1: bt_obs_m_cdf
}

riaa_f_cdf = {
    (10.0/90.0): 5,
    (40.0/90.0): 6,
    1.0:         7
}

riaa_m_cdf = {
    (10.0/90.0): 3,
    (30.0/90.0): 4,
    (60.0/90.0): 5,
    (80.0/90.0): 6,
    1.0:         7
}

riaa_map = {
    'female': riaa_f_cdf,
    'male': riaa_m_cdf
}

rila_f_cdf = {
    (10.0/90.0): 3,
    (30.0/90.0): 4,
    (60.0/90.0): 5,
    (80.0/90.0): 6,
    1.0:         7
}

rila_m_cdf = {
    (50.0/90.0): 3,
    (80.0/90.0): 4,
    1.0:         5
}

rila_map = {
    'female': rila_f_cdf,
    'male': rila_m_cdf
}

# The following tables use modifier values rather than absolute values.
outer_angle__modifier_cdf = {
    (30.0/110.0): -2,
    (50.0/110.0): -1,
    (60.0/110.0): 0,
    (80.0/110.0): 1,
    1.0:          2
}

obs_err_1_modifier_cdf = {
    0.7: 0,
    0.9: 1,
    1.0: 2
}

obs_err_2_modifier_cdf = {
    0.3: -1,
    0.7: 0,
    0.9: 1,
    1.0: 2
}

obs_err_typ_modifier_cdf = {
    0.1: -2,
    0.3: -1,
    0.7: 0,
    0.9: 1,
    1.0: 2
}

obs_err_8_modifier_cdf = {
    0.1: -2,
    0.3: -1,
    0.7: 0,
    1.0: 1
}

obs_err_9_modifier_cdf = {
    0.1: -2,
    0.3: -1,
    1.0: 0
}

def sample_distr(cdf_table):
    rand = rng.random()
    prev_key = 0.0
    for key in cdf_table:
        if rand >= prev_key and rand < key:
            return cdf_table[key]

def get_sex():
    return sample_distr(sex_cdf)

# Expects 'male' or 'female'
def get_bt(sex):
    return sex_bt_map[sex]

# Expects 0 (hexagon) or 1 (circle)
def get_obs_bt(body_type):
    return sample_distr(bt_obs_bt_map[body_type])

# Expects 'male' or 'female'
def get_riaa(sex):
    return sample_distr(riaa_map[sex])

# Expects 1-9
def get_roaa(riaa):
    return riaa + sample_distr(outer_angle__modifier_cdf)

# Expects 'male' or 'female'
def get_rila(sex):
    return sample_distr(rila_map[sex])

def get_rola(rila):
    return rila + sample_distr(outer_angle__modifier_cdf)

# Expected range: 1-9
def get_obs_angle_val(x):
    if x == 1:
        table = obs_err_1_modifier_cdf
    elif x == 2:
        table = obs_err_2_modifier_cdf
    elif x == 8:
        table = obs_err_8_modifier_cdf
    elif x == 9:
        table = obs_err_9_modifier_cdf
    else:
        table = obs_err_typ_modifier_cdf
    return x + sample_distr(table)

def flip_angle(x):
    return 9 - x + 1

def sample_model():
    sex = get_sex()
    bt = get_bt(sex)
    obs_bt = get_obs_bt(bt)

    riaa = get_riaa(sex)
    obs_riaa = get_obs_angle_val(riaa)
    liaa = flip_angle(riaa)
    obs_liaa = get_obs_angle_val(liaa)

    roaa = get_roaa(riaa)
    obs_roaa = get_obs_angle_val(roaa)
    loaa = flip_angle(roaa)
    obs_loaa = get_obs_angle_val(loaa)

    rila = get_rila(sex)
    obs_rila = get_obs_angle_val(rila)
    lila = flip_angle(rila)
    obs_lila = get_obs_angle_val(lila)

    rola = get_rola(rila)
    obs_rola = get_obs_angle_val(rola)
    lola = flip_angle(rola)
    obs_lola = get_obs_angle_val(lola)

    return {
        'sex': sex,
        'bt': bt,
        'obs_bt': obs_bt,
        'riaa': riaa,
        'obs_riaa': obs_riaa,
        'liaa': liaa,
        'obs_liaa': obs_liaa,
        'roaa': roaa,
        'obs_roaa': obs_roaa,
        'loaa': loaa,
        'obs_loaa': obs_loaa,
        'rila': rila,
        'obs_rila': obs_rila,
        'lila': lila,
        'obs_lila': obs_lila,
        'rola': rola,
        'obs_rola': obs_rola,
        'lola': lola,
        'obs_lola': obs_lola,
    }

'''
def connectNodes(valueNode, factorNode, valueNodeDimensionIndex):
    valueNode.connectNode(factorNode)
    factorNode.connectNode(valueNode, valueNodeDimensionIndex)

class Node:

    def __init__(self, name):
        self.name = name
        # List of value nodes
        self.connectedNodes = []
        # Should be the same size as connectedNodes
        self.savedOutgoingMessages = []

    def clearAllSavedMessages(self, excludedNode):
        for i in range(len(self.savedOutgoingMessages)):
            self.savedOutgoingMessages[i] = []
        for node in self.connectedNodes:
            if node != excludedNode:
                node.clearAllSavedMessages(self)

class FactorNode(Node):

    def __init__(self, name, factor):
        Node.__init__(self, name)
        self.nodeDimensionIndices = []
        self.factor = factor

    def connectNode(self, node, nodeDimensionIndex):
        self.connectedNodes.append(node)
        self.savedOutgoingMessages.append([])
        self.nodeDimensionIndices.append(nodeDimensionIndex)

    def setLeaf(self):
        for i in range(len(self.savedOutgoingMessages)):
            self.savedOutgoingMessages[i] = self.factor
            print(self.factor)

    def getMessage(self, targetNode):
        print(self.name + '.getMessage()')
        # Target node index is needed to save any computed values
        targetNodeIdx = -1
        for i in range(len(self.connectedNodes)):
            if self.connectedNodes[i] == targetNode:
                targetNodeIdx = i
                break
        # Need to do some computation if the message isn't saved
        if targetNodeIdx == -1 or len(self.savedOutgoingMessages[targetNodeIdx]) == 0:
            print(self.savedOutgoingMessages)
            accum1 = []
            # Product portion. This is similar to variable node.
            for connectedNode in self.connectedNodes:
                if connectedNode == targetNode:
                    continue
                if len(accum1) == 0:
                    accum1 = connectedNode.getMessage(self)
                else:
                    accum1 *= connectedNode.getMessage(self)
            # Sum portion. We sum over all bound variables (i.e. all possible 
            # values of the connected nodes excluding the target node)
            
            # Node: We make a simplifying assumption here that the factor nodes 
            # are two-dimensional. TODO doesn't work in general.
            accum2 = []
            if self.nodeDimensionIndices[targetNodeIdx] == 0:
                # Need to sum over columns
                for i in range(self.factor.shape[1]):
                    if len(accum2) == 0:
                        accum2 = self.factor[:, i]
                    else:
                        accum2 += self.factor[:, i]
            else:
                # Need to sum over rows
                for i in range(self.factor.shape[0]):
                    if len(accum2) == 0:
                        accum2 = self.factor[i]
                    else:
                        accum2 += self.factor[i]
            print(accum2)
            print(accum1)
            result = np.multiply(accum2, accum1)
            if targetNodeIdx != -1:
                self.savedOutgoingMessages[targetNodeIdx] = result
            print(self.name + ' sending message to ' + targetNode.name + ': ' + str(result))
            return result
        print(self.name + ' sending message to ' + targetNode.name + ': ' + str(self.savedOutgoingMessages[targetNodeIdx]))
        return self.savedOutgoingMessages[targetNodeIdx]

class ValueNode(Node):

    def __init__(self, name):
        Node.__init__(self, name)

    def connectNode(self, node):
        self.connectedNodes.append(node)
        self.savedOutgoingMessages.append([])

    def setLeaf(self):
        for i in range(len(self.savedOutgoingMessages)):
            self.savedOutgoingMessages[i] = 1

    def getMessage(self, targetNode):
        # Target node index is needed to save any computed values
        print(self.name + '.getMessage()')
        targetNodeIdx = -1
        for i in range(len(self.connectedNodes)):
            if self.connectedNodes[i] == targetNode:
                targetNodeIdx = i
                break
        print(targetNodeIdx)
        print(len(self.savedOutgoingMessages[targetNodeIdx]))
        if targetNodeIdx == -1 or len(self.savedOutgoingMessages[targetNodeIdx]) == 0:
            accum1 = []
            print(accum1)
            for connectedNode in self.connectedNodes:
                print('connectedNode: ' + connectedNode.name)
                print(accum1)
                if connectedNode == targetNode:
                    continue
                if len(accum1) == 0:
                    accum1 = connectedNode.getMessage(self)
                else:
                    # TODO should this be element-wise or matrix multiplication?
                    accum1 = np.multiply(accum1, connectedNode.getMessage(self))
            if targetNodeIdx != -1:
                self.savedOutgoingMessages[targetNodeIdx] = accum1
            print(self.name + ' sending message to ' + targetNode.name + ': ' + str(accum1))
            return accum1
        print(self.name + ' sending message to ' + targetNode.name + ': ' + str(self.savedOutgoingMessages[targetNodeIdx]))
        return self.savedOutgoingMessages[targetNodeIdx]

bt_to_array_index = {
    1: 0,
    0: 1
}
angle_to_array_index = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8
}

# MQL Factor Graph, which we KNOW is a tree
# For factor nodes, first dimension (row) will always be the variable, second 
# dimension (col) will always be the condition.

# Start with some shared arrays:
array_obs_given_real = np.zeros((9, 9))
array_obs_given_real[0, 0] = 0.7
array_obs_given_real[1, 0] = 0.2
array_obs_given_real[2, 0] = 0.1

array_obs_given_real[0, 1] = 0.3
array_obs_given_real[1, 1] = 0.4
array_obs_given_real[2, 1] = 0.2
array_obs_given_real[3, 1] = 0.1

for i in range(2, 7):
    array_obs_given_real[i - 2, i] = 0.1
    array_obs_given_real[i - 1, i] = 0.2
    array_obs_given_real[i,     i] = 0.4
    array_obs_given_real[i + 1, i] = 0.2
    array_obs_given_real[i + 2, i] = 0.1

array_obs_given_real[5, 7] = 0.1
array_obs_given_real[6, 7] = 0.2
array_obs_given_real[7, 7] = 0.4
array_obs_given_real[8, 7] = 0.3

array_obs_given_real[6, 8] = 0.1
array_obs_given_real[7, 8] = 0.2
array_obs_given_real[8, 8] = 0.7

array_flipped_obs_given_real = np.zeros((9, 9))
array_flipped_obs_given_real[6, 0] = 0.1
array_flipped_obs_given_real[7, 0] = 0.2
array_flipped_obs_given_real[8, 0] = 0.7

array_flipped_obs_given_real[5, 1] = 0.1
array_flipped_obs_given_real[6, 1] = 0.2
array_flipped_obs_given_real[7, 1] = 0.4
array_flipped_obs_given_real[8, 1] = 0.3

for i in range(2, 7):
    array_flipped_obs_given_real[8 - i - 2, i] = 0.1
    array_flipped_obs_given_real[8 - i - 1, i] = 0.2
    array_flipped_obs_given_real[8 - i,     i] = 0.4
    array_flipped_obs_given_real[8 - i + 1, i] = 0.2
    array_flipped_obs_given_real[8 - i + 2, i] = 0.1

array_flipped_obs_given_real[0, 7] = 0.3
array_flipped_obs_given_real[1, 7] = 0.4
array_flipped_obs_given_real[2, 7] = 0.2
array_flipped_obs_given_real[3, 7] = 0.1

array_flipped_obs_given_real[1, 8] = 0.2
array_flipped_obs_given_real[0, 8] = 0.7
array_flipped_obs_given_real[2, 8] = 0.1

array_outer_given_inner = np.zeros((9, 9))
for i in range(2, 7):
    array_outer_given_inner[i - 2, i] = (30.0/110.0)
    array_outer_given_inner[i - 1, i] = (20.0/110.0)
    array_outer_given_inner[i,     i] = (10.0/110.0)
    array_outer_given_inner[i + 1, i] = (20.0/110.0)
    array_outer_given_inner[i + 2, i] = (30.0/110.0)

# Now, to construct the factor graph.
fact_bt = FactorNode('fact_bt', np.array([(2.0/3.0), (1.0/3.0)]))
fact_bt.setLeaf()

val_bt = ValueNode('val_bt')
connectNodes(val_bt, fact_bt, 0)

fact_riaa_given_bt = FactorNode('fact_riaa_given_bt', np.array(
   [[0.0,         0.0],
    [0.0,         0.0],
    [(10.0/90.0), 0.0],
    [(20.0/90.0), 0.0],
    [(30.0/90.0), (10.0/90.0)],
    [(20.0/90.0), (30.0/90.0)],
    [(20.0/90.0), (60.0/90.0)],
    [0.0,         0.0],
    [0.0,         0.0]]))
connectNodes(val_bt, fact_riaa_given_bt, 1)

val_riaa = ValueNode('val_riaa')
connectNodes(val_riaa, fact_riaa_given_bt, 0)

fact_oriaa_given_riaa = FactorNode('fact_oriaa_given_riaa', array_obs_given_real)
connectNodes(val_riaa, fact_oriaa_given_riaa, 1)

val_oriaa = ValueNode('val_oriaa')
connectNodes(val_oriaa, fact_oriaa_given_riaa, 0)

fact_oriaa = FactorNode('fact_oriaa', None)
connectNodes(val_oriaa, fact_oriaa, 0)
fact_oriaa.setLeaf()

fact_oliaa_given_riaa = FactorNode('fact_oliaa_given_riaa', array_flipped_obs_given_real)
connectNodes(val_riaa, fact_oliaa_given_riaa, 1)

val_oliaa = ValueNode('val_oliaa')
connectNodes(val_oliaa, fact_oliaa_given_riaa, 0)

fact_oliaa = FactorNode('fact_oliaa', None)
connectNodes(val_oliaa, fact_oliaa, 0)
fact_oliaa.setLeaf()

fact_roaa_given_riaa = FactorNode('fact_roaa_given_riaa', array_outer_given_inner)
connectNodes(val_riaa, fact_roaa_given_riaa, 1)

val_roaa = ValueNode('val_roaa')
connectNodes(val_roaa, fact_roaa_given_riaa, 0)

fact_oroaa_given_roaa = FactorNode('fact_oroaa_given_roaa', array_obs_given_real)
connectNodes(val_roaa, fact_oroaa_given_roaa, 1)

val_oroaa = ValueNode('val_oroaa')
connectNodes(val_oroaa, fact_oroaa_given_roaa, 0)

fact_oroaa = FactorNode('fact_oroaa', None)
connectNodes(val_oroaa, fact_oroaa, 0)
fact_oroaa.setLeaf()

fact_oloaa_given_roaa = FactorNode('fact_oloaa_given_roaa', array_flipped_obs_given_real)
connectNodes(val_roaa, fact_oloaa_given_roaa, 1)

val_oloaa = ValueNode('val_oloaa')
connectNodes(val_oloaa, fact_oloaa_given_roaa, 0)

fact_oloaa = FactorNode('fact_oloaa', None)
connectNodes(val_oloaa, fact_oloaa, 0)
fact_oloaa.setLeaf()

fact_obt_given_bt = FactorNode('fact_obt_given_bt', np.array(
   [[0.7, 0.3],
    [0.3, 0.7]]))
connectNodes(val_bt, fact_obt_given_bt, 1)

val_obt = ValueNode('val_obt')
connectNodes(val_obt, fact_obt_given_bt, 0)

fact_obt = FactorNode('fact_obt', None)
connectNodes(val_obt, fact_obt, 0)
fact_obt.setLeaf()

fact_rila_given_bt = FactorNode('fact_rila_given_bt', np.array(
   [[0.0,        0.0],
    [0.0,         0.0],
    [(50.0/90.0), (10.0/90.0)],
    [(30.0/90.0), (20.0/90.0)],
    [(10.0/90.0), (30.0/90.0)],
    [0.0,         (20.0/90.0)],
    [0.0,         (10.0/90.0)],
    [0.0,         0.0],
    [0.0,         0.0]]))
connectNodes(val_bt, fact_rila_given_bt, 1)

val_rila = ValueNode('val_rila')
connectNodes(val_rila, fact_rila_given_bt, 0)

fact_orila_given_rila = FactorNode('fact_orila_given_rila', array_obs_given_real)
connectNodes(val_rila, fact_orila_given_rila, 1)

val_orila = ValueNode('val_orila')
connectNodes(val_orila, fact_orila_given_rila, 0)

fact_orila = FactorNode('fact_orila', None)
connectNodes(val_orila, fact_orila, 0)
fact_orila.setLeaf()

fact_olila_given_rila = FactorNode('fact_olila_given_rila', array_flipped_obs_given_real)
connectNodes(val_rila, fact_olila_given_rila, 1)

val_olila = ValueNode('val_olila')
connectNodes(val_olila, fact_olila_given_rila, 0)

fact_olila = FactorNode('fact_olila', None)
connectNodes(val_olila, fact_olila, 0)
fact_olila.setLeaf()

fact_rola_given_rila = FactorNode('fact_rola_given_rila', array_outer_given_inner)
connectNodes(val_rila, fact_rola_given_rila, 1)

val_rola = ValueNode('val_rola')
connectNodes(val_rola, fact_rola_given_rila, 0)

fact_orola_given_rola = FactorNode('fact_orola_given_rola', array_obs_given_real)
connectNodes(val_rola, fact_orola_given_rola, 1)

val_orola = ValueNode('val_orola')
connectNodes(val_orola, fact_orola_given_rola, 0)

fact_orola = FactorNode('fact_orola', None)
connectNodes(val_orola, fact_orola, 0)
fact_orola.setLeaf()

fact_olola_given_rola = FactorNode('fact_olola_given_rola', array_flipped_obs_given_real)
connectNodes(val_rola, fact_olola_given_rola, 1)

val_olola = ValueNode('val_olola')
connectNodes(val_olola, fact_olola_given_rola, 0)

fact_olola = FactorNode('fact_olola', None)
connectNodes(val_olola, fact_olola, 0)
fact_olola.setLeaf()

for i in range(1000):
    observation = sample_model()

    ary_obt = np.zeros((2, ))
    ary_obt[bt_to_array_index[observation['obs_bt']]] = 1.0
    fact_obt.factor = ary_obt

    ary_oriaa = np.zeros((9, ))
    ary_oriaa[angle_to_array_index[observation['obs_riaa']]] = 1.0
    fact_oriaa.factor = ary_oriaa

    ary_oliaa = np.zeros((9, ))
    ary_oliaa[angle_to_array_index[observation['obs_liaa']]] = 1.0
    fact_oliaa.factor = ary_oliaa

    ary_orila = np.zeros((9, ))
    ary_orila[angle_to_array_index[observation['obs_rila']]] = 1.0
    fact_orila.factor = ary_orila

    ary_olila = np.zeros((9, ))
    ary_olila[angle_to_array_index[observation['obs_lila']]] = 1.0
    fact_olila.factor = ary_olila

    ary_oroaa = np.zeros((9, ))
    ary_oroaa[angle_to_array_index[observation['obs_roaa']]] = 1.0
    fact_oroaa.factor = ary_oroaa

    ary_oloaa = np.zeros((9, ))
    ary_oloaa[angle_to_array_index[observation['obs_loaa']]] = 1.0
    fact_oloaa.factor = ary_oloaa

    ary_orola = np.zeros((9, ))
    ary_orola[angle_to_array_index[observation['obs_rola']]] = 1.0
    fact_orola.factor = ary_orola

    ary_olola = np.zeros((9, ))
    ary_olola[angle_to_array_index[observation['obs_lola']]] = 1.0
    fact_olola.factor = ary_olola

    val_bt.clearAllSavedMessages(None)
    # Need to call this because of poor architecture
    fact_bt.setLeaf()
    fact_obt.setLeaf()
    fact_oriaa.setLeaf()
    fact_oliaa.setLeaf()
    fact_orila.setLeaf()
    fact_olila.setLeaf()
    fact_oroaa.setLeaf()
    fact_oloaa.setLeaf()
    fact_orola.setLeaf()
    fact_olola.setLeaf()

    print(val_bt.getMessage(None))
'''
array_bt = np.array([(1.0/3.0), (2.0/3.0)])

array_obt_given_bt = np.array(
   [[0.7, 0.3],
    [0.3, 0.7]])

array_obs_given_real = np.zeros((9, 9))
array_obs_given_real[0, 0] = 0.7
array_obs_given_real[0, 1] = 0.2
array_obs_given_real[0, 2] = 0.1

array_obs_given_real[1, 0] = 0.3
array_obs_given_real[1, 1] = 0.4
array_obs_given_real[1, 2] = 0.2
array_obs_given_real[1, 3] = 0.1

for i in range(2, 7):
    array_obs_given_real[i, i-2] = 0.1
    array_obs_given_real[i, i-1] = 0.2
    array_obs_given_real[i, i] = 0.4
    array_obs_given_real[i, i+1] = 0.2
    array_obs_given_real[i, i+2] = 0.1

array_obs_given_real[7, 5] = 0.1
array_obs_given_real[7, 6] = 0.2
array_obs_given_real[7, 7] = 0.4
array_obs_given_real[7, 8] = 0.3

array_obs_given_real[8, 6] = 0.1
array_obs_given_real[8, 7] = 0.2
array_obs_given_real[8, 8] = 0.7

print(array_obs_given_real)

array_flipped_obs_given_real = np.zeros((9, 9))
array_flipped_obs_given_real[0, 6] = 0.1
array_flipped_obs_given_real[0, 7] = 0.2
array_flipped_obs_given_real[0, 8] = 0.7

array_flipped_obs_given_real[1, 5] = 0.1
array_flipped_obs_given_real[1, 6] = 0.2
array_flipped_obs_given_real[1, 7] = 0.4
array_flipped_obs_given_real[1, 8] = 0.3

for i in range(2, 7):
    array_flipped_obs_given_real[i, 8-i-2] = 0.1
    array_flipped_obs_given_real[i, 8-i-1] = 0.2
    array_flipped_obs_given_real[i, 8-i] = 0.4
    array_flipped_obs_given_real[i, 8-i+1] = 0.2
    array_flipped_obs_given_real[i, 8-i+2] = 0.1

array_flipped_obs_given_real[7, 0] = 0.3
array_flipped_obs_given_real[7, 1] = 0.4
array_flipped_obs_given_real[7, 2] = 0.2
array_flipped_obs_given_real[7, 3] = 0.1

array_flipped_obs_given_real[8, 0] = 0.7
array_flipped_obs_given_real[8, 1] = 0.2
array_flipped_obs_given_real[8, 2] = 0.1

print(array_flipped_obs_given_real)

array_outer_given_inner = np.zeros((9, 9))
for i in range(2, 7):
    array_outer_given_inner[i, i-2] = (30.0/110.0)
    array_outer_given_inner[i, i-1] = (20.0/110.0)
    array_outer_given_inner[i, i] = (10.0/110.0)
    array_outer_given_inner[i, i+1] = (20.0/110.0)
    array_outer_given_inner[i, i+2] = (30.0/110.0)

print(array_outer_given_inner)

array_riaa_given_bt = np.array([
    [0.0, 0.0, 0.0,       0.0,       10.0/90.0, 30.0/90.0, 50.0/90.0, 0.0, 0.0],
    [0.0, 0.0, 10.0/90.0, 20.0/90.0, 30.0/90.0, 20.0/90.0, 10.0/90.0, 0.0, 0.0]])

print(array_riaa_given_bt)

array_rila_given_bt = np.array([
    [0.0, 0.0, 10.0/90.0, 20.0/90.0, 30.0/90.0, 20.0/90.0, 10.0/90.0, 0.0, 0.0],
    [0.0, 0.0, 50.0/90.0, 30.0/90.0, 10.0/90.0, 0.0,       0.0,       0.0, 0.0]])

print(array_rila_given_bt)

# It would probably be better not to hard-code the structure of the tree. I 
# spent many hours trying to produce a general tree solution, but after many 
# hours of working through it and debugging, I was spinning my wheels. I 
# realized hard-coding things here 
# would actually help me better understand the process of the algorithm, in 
# general.
def get_sum_product_preds(obt, oriaa, oliaa, orila, olila, oroaa, oloaa, orola, olola):
    # Convert values to one-hot vectors
    obt_vec = np.zeros((2, ))
    obt_vec[obt] = 1.0
    oriaa_vec = np.zeros((9, ))
    oriaa_vec[oriaa - 1] = 1.0
    oliaa_vec = np.zeros((9, ))
    oliaa_vec[oliaa - 1] = 1.0
    orila_vec = np.zeros((9, ))
    orila_vec[orila - 1] = 1.0
    olila_vec = np.zeros((9, ))
    olila_vec[olila - 1] = 1.0
    oroaa_vec = np.zeros((9, ))
    oroaa_vec[oroaa - 1] = 1.0
    oloaa_vec = np.zeros((9, ))
    oloaa_vec[oloaa - 1] = 1.0
    orola_vec = np.zeros((9, ))
    orola_vec[orola - 1] = 1.0
    olola_vec = np.zeros((9, ))
    olola_vec[olola - 1] = 1.0
    
    obt_msg = array_obt_given_bt.dot(obt_vec)
    oriaa_msg = array_obs_given_real.dot(oriaa_vec)
    oliaa_msg = array_flipped_obs_given_real.dot(oliaa_vec)
    orila_msg = array_obs_given_real.dot(orila_vec)
    olila_msg = array_flipped_obs_given_real.dot(olila_vec)
    oroaa_msg = array_obs_given_real.dot(oroaa_vec)
    oloaa_msg = array_flipped_obs_given_real.dot(oloaa_vec)
    orola_msg = array_obs_given_real.dot(orola_vec)
    olola_msg = array_flipped_obs_given_real.dot(olola_vec)
    
    # When combining multiple sources, first multiply them together, then normalize (equivalent to the value node operation).
    # Finally, feed it through the matrix.
    roaa_msg = oroaa_msg * oloaa_msg
    roaa_msg = array_outer_given_inner.dot(roaa_msg)
    roaa_msg = roaa_msg / np.sum(roaa_msg)
    rola_msg = orola_msg * olola_msg
    rola_msg = array_outer_given_inner.dot(rola_msg)
    rola_msg = rola_msg / np.sum(rola_msg)

    riaa_msg = oriaa_msg * oliaa_msg * roaa_msg
    riaa_msg = array_riaa_given_bt.dot(riaa_msg)
    riaa_msg = riaa_msg / np.sum(riaa_msg)
    rila_msg = orila_msg * olila_msg * rola_msg
    rila_msg = array_rila_given_bt.dot(rila_msg)
    rila_msg = rila_msg / np.sum(rila_msg)

    # Need to also incorporate the prior, here.
    bt_msg = obt_msg * riaa_msg * rila_msg * array_bt
    bt_msg = bt_msg / np.sum(bt_msg)
    return bt_msg

def get_max_sum_preds(obt, oriaa, oliaa, orila, olila, oroaa, oloaa, orola, olola):
    # Convert values to one-hot vectors
    obt_vec = np.full((2, ), -np.inf)
    obt_vec[obt] = 0.0
    oriaa_vec = np.full((9, ), -np.inf)
    oriaa_vec[oriaa - 1] = 0.0
    oliaa_vec = np.full((9, ), -np.inf)
    oliaa_vec[oliaa - 1] = 0.0
    orila_vec = np.full((9, ), -np.inf)
    orila_vec[orila - 1] = 0.0
    olila_vec = np.full((9, ), -np.inf)
    olila_vec[olila - 1] = 0.0
    oroaa_vec = np.full((9, ), -np.inf)
    oroaa_vec[oroaa - 1] = 0.0
    oloaa_vec = np.full((9, ), -np.inf)
    oloaa_vec[oloaa - 1] = 0.0
    orola_vec = np.full((9, ), -np.inf)
    orola_vec[orola - 1] = 0.0
    olola_vec = np.full((9, ), -np.inf)
    olola_vec[olola - 1] = 0.0
    
    # Leaf nodes
    # For sum-product above, the matrix multiplication ('.dot') combined the 
    # multiplication step and the marginalization step. We may need to break 
    # that up here.
    obt_msg = np.log(array_obt_given_bt) + obt_vec
    obt_max_idx = np.argmax(obt_msg, axis = 1)
    obt_msg = np.max(obt_msg, axis = 1)

    oriaa_msg = np.log(array_obs_given_real) + oriaa_vec
    oriaa_max_idx = np.argmax(oriaa_msg, axis = 1)
    oriaa_msg = np.max(oriaa_msg, axis = 1)

    oliaa_msg = np.log(array_flipped_obs_given_real) + oliaa_vec
    oliaa_max_idx = np.argmax(oliaa_msg, axis = 1)
    oliaa_msg = np.max(oliaa_msg, axis = 1)

    orila_msg = np.log(array_obs_given_real) + orila_vec
    orila_max_idx = np.argmax(orila_msg, axis = 1)
    orila_msg = np.max(orila_msg, axis = 1)

    olila_msg = np.log(array_flipped_obs_given_real) + olila_vec
    olila_max_idx = np.argmax(olila_msg, axis = 1)
    olila_msg = np.max(olila_msg, axis = 1)

    oroaa_msg = np.log(array_obs_given_real) + oroaa_vec
    oroaa_max_idx = np.argmax(oroaa_msg, axis = 1)
    oroaa_msg = np.max(oroaa_msg, axis = 1)

    oloaa_msg = np.log(array_flipped_obs_given_real) + oloaa_vec
    oloaa_max_idx = np.argmax(oloaa_msg, axis = 1)
    oloaa_msg = np.max(oloaa_msg, axis = 1)

    orola_msg = np.log(array_obs_given_real) + orola_vec
    orola_max_idx = np.argmax(orola_msg, axis = 1)
    orola_msg = np.max(orola_msg, axis = 1)

    olola_msg = np.log(array_flipped_obs_given_real) + olola_vec
    olola_max_idx = np.argmax(olola_msg, axis = 1)
    olola_msg = np.max(olola_msg, axis = 1)

    ### More complicated nodes
    roaa_msg = oroaa_msg + oloaa_msg
    roaa_msg = np.log(array_outer_given_inner) + roaa_msg
    roaa_max_idx = np.argmax(roaa_msg, axis = 1)
    roaa_msg = np.max(roaa_msg, axis = 1)

    rola_msg = orola_msg + olola_msg
    rola_msg = np.log(array_outer_given_inner) + rola_msg
    rola_max_idx = np.argmax(rola_msg, axis = 1)
    rola_msg = np.max(rola_msg, axis = 1)

    riaa_msg = oriaa_msg + oliaa_msg + roaa_msg
    riaa_msg = np.log(array_riaa_given_bt) + riaa_msg
    riaa_max_idx = np.argmax(riaa_msg, axis = 1)
    riaa_msg = np.max(riaa_msg, axis = 1)

    rila_msg = orila_msg + olila_msg + rola_msg
    rila_msg = np.log(array_rila_given_bt) + rila_msg
    rila_max_idx = np.argmax(rila_msg, axis = 1)
    rila_msg = np.max(rila_msg, axis = 1)

    # Need to also incorporate the prior, here.
    bt_msg = obt_msg + riaa_msg + rila_msg + array_bt
    bt_max_idx = np.argmax(bt_msg)

    sex = 'female' if bt_max_idx == 0 else 'male'
    bt = bt_max_idx
    riaa = riaa_max_idx[bt_max_idx] + 1
    liaa = flip_angle(riaa)
    roaa = roaa_max_idx[riaa - 1] + 1
    loaa = flip_angle(roaa)
    rila = rila_max_idx[bt_max_idx] + 1
    lila = flip_angle(rila)
    rola = rola_max_idx[rila - 1] + 1
    lola = flip_angle(rola)
    return {
        'sex': sex,
        'bt': bt,
        'obs_bt': obt,
        'riaa': riaa,
        'obs_riaa': oriaa,
        'liaa': liaa,
        'obs_liaa': oliaa,
        'roaa': roaa,
        'obs_roaa': oroaa,
        'loaa': loaa,
        'obs_loaa': oloaa,
        'rila': rila,
        'obs_rila': orila,
        'lila': lila,
        'obs_lila': olila,
        'rola': rola,
        'obs_rola': orola,
        'lola': lola,
        'obs_lola': olola,
    }

if __name__ == '__main__':
    rng.seed(42)
    num_trials = 100000
    correct_predictions = 0
    correct_max_sum_preds = 0
    class_label_bins = [[], [], [], [], [], [], [], [], [], []]
    for i in range(num_trials):
        trial = sample_model()
        #print(trial)
        pred_cur_vec = get_sum_product_preds(
            trial['obs_bt'],
            trial['obs_riaa'],
            trial['obs_liaa'],
            trial['obs_rila'],
            trial['obs_lila'],
            trial['obs_roaa'],
            trial['obs_loaa'],
            trial['obs_rola'],
            trial['obs_lola'])
        if pred_cur_vec[0] > pred_cur_vec[1] and trial['bt'] == 0:
            correct_predictions += 1
        elif pred_cur_vec[0] <= pred_cur_vec[1] and trial['bt'] == 1:
            correct_predictions += 1

        # Bin for Q2
        which_bin = int(pred_cur_vec[1] * 10)
        # This will only happen if the sum-product is 100% sure.
        if which_bin == 10:
            which_bin = 9
        class_label_bins[which_bin].append(trial['bt'])

        # Max-sum for Q3
        max_config = get_max_sum_preds(
            trial['obs_bt'],
            trial['obs_riaa'],
            trial['obs_liaa'],
            trial['obs_rila'],
            trial['obs_lila'],
            trial['obs_roaa'],
            trial['obs_loaa'],
            trial['obs_rola'],
            trial['obs_lola'])
        if max_config['bt'] == trial['bt']:
            correct_max_sum_preds += 1
        elif max_config['bt'] == trial['bt']:
            correct_max_sum_preds += 1

    print('Sum Product Accuracy: ' + str(correct_predictions / num_trials))
    print('Max Sum Accuracy: ' + str(correct_max_sum_preds / num_trials))

    # More calculations for Q2
    male_percentage_bins = []
    for class_label_bin in class_label_bins:
        #print(class_label_bin)
        if len(class_label_bin) == 0:
            to_append = 0.0
        else:
            to_append = np.sum(np.array(class_label_bin)) / len(class_label_bin)
        male_percentage_bins.append(to_append)
    male_percentage_X = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

    fig, ax = plt.subplots()
    rects = ax.bar(male_percentage_X, male_percentage_bins, 10.0)
    ax.set_title('Percentage of Actual Males as a Function of Predicted Male Probability')
    ax.set_xlabel('Predicted Male Probability Bins')
    ax.set_ylabel('Percentage of Actual Males in the Bin')
    plt.show()

    for i in range(10):
        trial = sample_model()
        max_config = get_max_sum_preds(
            trial['obs_bt'],
            trial['obs_riaa'],
            trial['obs_liaa'],
            trial['obs_rila'],
            trial['obs_lila'],
            trial['obs_roaa'],
            trial['obs_loaa'],
            trial['obs_rola'],
            trial['obs_lola'])
        draw_mql(9, 15, trial['obs_bt'], trial['obs_riaa'], trial['obs_rila'], trial['obs_liaa'], trial['obs_lila'], trial['obs_roaa'], trial['obs_rola'], trial['obs_loaa'], trial['obs_lola'])
        draw_mql(9, 15, max_config['bt'], max_config['riaa'], max_config['rila'], max_config['liaa'], max_config['lila'], max_config['roaa'], max_config['rola'], max_config['loaa'], max_config['lola'])
        draw_mql(9, 15, trial['bt'], trial['riaa'], trial['rila'], trial['liaa'], trial['lila'], trial['roaa'], trial['rola'], trial['loaa'], trial['lola'])
