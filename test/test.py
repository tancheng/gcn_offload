#!/usr/bin/env python
#=========================================================================
# test.py
#=========================================================================
#
# Author : Cheng Tan
# Date   : Oct 3, 2020

import argparse
import os
import sys
import time
from collections import deque

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import ctypes
#from torch_geometric.datasets import Planetoid

class custom_python:
  def mm(self, x, weight):
    out = [[0 for _ in weight[0]] for _ in x]
  #  trace("out: ", len(out), len(out[0]))
    for i in range(len(x)):
      out[i] = np.dot(x[i], weight)
#      for j in range(len(weight[0])):
#        out[i][j] = np.dot(x[i], weight[:,j])
#        for k in range(len(x[0])):
#          out[i][j] += x[i][k] * weight[k][j]
  #  A = np.array(x)
  #  B = np.array(weight)
  #  out = A.dot(B)
    return out
  
  def add(self, x, bias):
    for i in range(len(x)):
      for j in range(len(x[0])):
        x[i][j] += bias[j]
    return x
        
  def relu(self, x):
    for i in range(len(x)):
      for j in range(len(x[0])):
        if x[i][j] < 0:
          x[i][j] = 0
    return x
   
  def buildEdge(self, edge, m, n):
    out = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(len(edge[0])):
      out[edge[0][i]][edge[1][i]] = 1
    return out

class custom_c:
  def mm(self, a, b):
    libmatmult = ctypes.CDLL("./mm.so")

    dima = len(a) * len(a[0])
    dimb = len(b) * len(b[0])
    dimc = len(a) * len(b[0])

    array_a = ctypes.c_float * dima
    array_b = ctypes.c_float * dimb
    array_c = ctypes.c_float * dimc

    suma = array_a()
    sumb = array_b()
    sumc = array_c()

    inda = 0
    for i in range(0, len(a)):
        for j in range(0, len(a[i])):
            suma[inda] = a[i][j]
            inda = inda + 1
    indb = 0
    for j in range(0, len(b[0])):
        for i in range(0, len(b)):
            sumb[indb] = b[i][j]
            indb = indb + 1

    libmatmult.matmult(ctypes.byref(suma), ctypes.byref(sumb), ctypes.byref(sumc), len(a), len(b[0]), len(b));

    res = np.zeros([len(a), len(b[0])])
    indc = 0
    for i in range(0, len(sumc)):
        res[indc][i % len(b[0])] = sumc[i]
        if i % len(b[0]) == len(b[0]) - 1:
            indc = indc + 1

    return res
  
#-------------------------------------------------------------------------
# verify the output of the customized execution with the golden model
#-------------------------------------------------------------------------

def verify(a, b):
  if len(a) != len(b):
    return False
  if len(a[0]) != len(b[0]):
    return False
  for i in range(len(a)):
    for j in range(len(a[0])):
      if abs(a[i][j] - b[i][j]) > 0.001:
        trace(i, j, a[i][j], b[i][j])
        return False
  return True

#-------------------------------------------------------------------------
# trace output in each step
#-------------------------------------------------------------------------

def trace(*info):
  print(*info)

#class Net(torch.nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.conv1 = GCNConv(dataset.num_node_features, 16, normalize=False)
#        self.conv2 = GCNConv(16, dataset.num_classes, normalize=False)
#
#    def forward(self, data):
#
#        trace("------------------------------- init start -------------------------------------")
#        start_time = time.time()
#        x, edge_index = data.x, data.edge_index
#        trace("[gold] feature ", len(x), " x ", len(x[0]), ": ")
#        trace(x)
#        trace("-------------------------------- init done -------------------------------------")
#        trace()
#
#        trace("------------------------------- conv1 start ------------------------------------")
#        start_time = time.time()
#        x = self.conv1(x, edge_index)
#        print("*** [gold] mm: %s seconds ***" % round(time.time() - start_time, 2))
#        trace("[gold] conv1 result ", len(x), " x ", len(x[0]), ": ")
#        trace(x)
#        trace("-------------------------------- conv1 done ------------------------------------")
#        trace()
#
#        trace("------------------------------- relu start -------------------------------------")
#        x = F.relu(x)
#        trace("[gold] relu: ")
#        trace(x)
#        trace("-------------------------------- relu done -------------------------------------")
#        trace()
#
#        trace("------------------------------- conv2 start ------------------------------------")
#        x = F.dropout(x, training=self.training)
#        x = self.conv2(x, edge_index)
#        trace("[gold] out (", len(x), "x", len(x[0]), "):")
#        trace(x)
#        trace("-------------------------------- conv2 done ------------------------------------")
#        print("*** [gold] entire GCN: %s seconds ***" % round(time.time() - start_time, 2))
#        trace()
#
#        # essential for training
#        # return F.log_softmax(x, dim=1)
#        return x

def main():
    a = np.random.normal(size=(800, 800)).astype('float32')
    b = np.random.normal(size=(800, 800)).astype('float32')
    custom = custom_python()
    print("============================== custom python matmul  ===================================")
    start_time = time.time()
    mm = custom.mm(a, b)
    print("*** [custom Python] mm: %s seconds ***" % round(time.time() - start_time, 4))
    trace()
#    print(mm)
#    trace("-------------------------- alternative conv1 start -----------------------------")
#    trace("[custom] alternative conv1 mm -- (A * X) * W: ")
#    mm = custom.mm(edgeMatrix, data.x)
#    mm = custom.mm(mm, weight1)
#    mm = custom.add(mm, bias1)
#    trace(mm)
#    trace("--------------------------- alternative conv1 done -----------------------------")
#    trace()

    custom = custom_c()
    print("============================== custom C matmul  ===================================")
    start_time = time.time()
    mm = custom.mm(a, b)
    print("*** [custom C outside lib] mm: %s seconds ***" % round(time.time() - start_time, 4))
    trace()
#    print(mm)

    print("============================== np matmul  ===================================")
    start_time = time.time()
    out1 = np.matmul(a, b)
    print("*** [numpy] mm: %s seconds ***" % round(time.time() - start_time, 4))

    print("============================== torch matmul  ===================================")
    start_time = time.time()
    out2 = torch.matmul(torch.tensor(a), torch.tensor(b))
    print("*** [torch] mm: %s seconds ***" % round(time.time() - start_time, 4))

    # verify the customized output with the golden model
    if verify(mm, out1) and verify(mm, out2):
      trace("[offload] success!")
    else:
      trace("[offload] fail!")

main()
