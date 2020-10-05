#!/usr/bin/env python
#=========================================================================
# netsim.py [options]
#=========================================================================
#
#  -h --help           Display this message
#  -v --verbose        Verbose mode
#
#  --offload-mode <pattern> Choose a offload mode for the GCN inference
#                           python : use the customized python code
#                           c++    : use the customized c++ code
#                           mpi    : distribute with the customized mpi code
#                           arena  : distribute with the ARENA library
#                           cgra   : offload compute to CGRA for acceleration
#
#  --trace             Display line-trace
#
# Graph Nerual Network. Choose an offload computation mode to execute.
#
# Author : Cheng Tan
# Date   : Oct 3, 2020

# Hack to add project root to python path

import argparse
import os
import sys
import time
from collections import deque

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from torch_geometric.datasets import Planetoid

#-------------------------------------------------------------------------
# Command line processing
#-------------------------------------------------------------------------

class ArgumentParserWithCustomError(argparse.ArgumentParser):
  def error( s, msg = "" ):
    if ( msg ): trace("\n ERROR: %s" % msg)
    trace("")
    file = open( sys.argv[0] )
    for ( lineno, line ) in enumerate( file ):
      if ( line[0] != '#' ): sys.exit(msg != "")
      if ( (lineno == 2) or (lineno >= 4) ): trace( line[1:].rstrip("\n") )

def parse_cmdline():
  parser = ArgumentParserWithCustomError( add_help=False )

  # Standard command line arguments

  parser.add_argument( "-v",
                       "--verbose",
                       action  = "store_true"                              )

  parser.add_argument( "-h",
                       "--help",
                       action = "store_true"                               )

  parser.add_argument( '--train',
                       action = 'store_true', default = False,
                       help = 'train network'                              )

  parser.add_argument( "--offload-mode",
                       choices = ["python", "c++", "mpi", "arena", "cgra"],
                       default = "python"                                  )

  parser.add_argument( "--trace",
                       action  = "store_true",
                       default = False                                     )

  opts = parser.parse_args()
  if opts.help: parser.error()
  return opts

dataset = Planetoid(root='/tmp/Cora', name='Cora')

#-------------------------------------------------------------------------
# customized python kernel preparing for offloading
#-------------------------------------------------------------------------

class custom_python:
  def mm(self, x, weight):
  #  out = [[0 for _ in weight[0]] for _ in x]
  #  trace("out: ", len(out), len(out[0]))
  #  for i in range(len(x)):
  #    for j in range(len(weight[0])):
  #      for k in range(len(x[0])):
  #        out[i][j] += x[i][k] * weight[k][j]
    A = np.array(x)
    B = np.array(weight)
    out = A.dot(B)
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
  opts = parse_cmdline()
  if opts.trace:
    print(*info)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16, normalize=False)
        self.conv2 = GCNConv(16, dataset.num_classes, normalize=False)

    def forward(self, data):

        trace("------------------------------- init start -------------------------------------")
        start_time = time.time()
        x, edge_index = data.x, data.edge_index
        trace("[gold] feature ", len(x), " x ", len(x[0]), ": ")
        trace(x)
        trace("-------------------------------- init done -------------------------------------")
        trace()

        trace("------------------------------- conv1 start ------------------------------------")
        x = self.conv1(x, edge_index)
        trace("[gold] conv1 result ", len(x), " x ", len(x[0]), ": ")
        trace(x)
        trace("-------------------------------- conv1 done ------------------------------------")
        trace()

        trace("------------------------------- relu start -------------------------------------")
        x = F.relu(x)
        trace("[gold] relu: ")
        trace(x)
        trace("-------------------------------- relu done -------------------------------------")
        trace()

        trace("------------------------------- conv2 start ------------------------------------")
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        trace("[gold] out (", len(x), "x", len(x[0]), "):")
        trace(x)
        trace("-------------------------------- conv2 done ------------------------------------")
        print("*** %s seconds ***" % round(time.time() - start_time, 2))
        trace()

        # essential for training
        # return F.log_softmax(x, dim=1)
        return x

def main():
  #parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  #args = parser.parse_args()
  PATH = "./weights.pt"
  
  opts = parse_cmdline()
  print("offload mode: ", opts.offload_mode, "; train: ", opts.train)
  
  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  device = torch.device('cpu')
  model = Net().to(device)
  data = dataset[0].to(device)
  
  if opts.train:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
  
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
  
    torch.save(model.state_dict(), PATH)
    # trace(model.state_dict())
    trace("[gold] train GCN and store weights into ", PATH)
  
  else:
    trace()
    print("=============================== custom offload =================================") 
    trace()
    trace("------------------------------ load weight start -------------------------------") 
    model.load_state_dict(torch.load(PATH))
    trace("[custom] load weights: ")
    # trace( model.state_dict())
    weight1 = model.state_dict()["conv1.weight"]
    bias1 = model.state_dict()["conv1.bias"]
    weight2 = model.state_dict()["conv2.weight"]
    bias2 = model.state_dict()["conv2.bias"]
    # trace("[custom] weight1 ", len(model.state_dict()["conv1.weight"].numpy()), " x ", len(model.state_dict()["conv1.weight"][0].numpy()), ": ")
    # trace(weight1)
    trace("----------------------------- weight loading done ------------------------------")
    trace()
  
    # Custom computation for offload and verification.
    if opts.offload_mode == "python":
      custom = custom_python()
    elif opts.offload_mode == "c++":
      custom = custom_c()
    elif opts.offload_mode == "arena":
      custom = custom_arena()
    elif opts.offload_mode == "cgra":
      custom = custom_cgra()
    else:
      trace("invalid offload mode")
      return

#    trace("--------------------------- build edge matrix start ----------------------------")
    edgeMatrix = custom.buildEdge(data.edge_index, len(data.x), len(data.x))
#    trace("---------------------------- build edge matrix done ----------------------------")
    trace("-------------------------------- conv1 start -----------------------------------")
    trace("[custom] conv1 mm -- A * (X * W): ")
    start_time = time.time()
    mm = custom.mm(data.x, weight1)
    # trace("..see mm: ")
    # trace(mm)
    mm = custom.mm(edgeMatrix, mm)
    mm = custom.add(mm, bias1)
    # trace("..final: ")
    trace(mm)
    trace("---------------------------------- conv1 done ----------------------------------")
    trace()
#    trace("-------------------------- alternative conv1 start -----------------------------")
#    trace("[custom] alternative conv1 mm -- (A * X) * W: ")
#    mm = custom.mm(edgeMatrix, data.x)
#    mm = custom.mm(mm, weight1)
#    mm = custom.add(mm, bias1)
#    trace(mm)
#    trace("--------------------------- alternative conv1 done -----------------------------")
#    trace()
    trace("------------------------------- relu start -------------------------------------")
    trace("[custom] custom relu: ")
    mm = custom.relu(mm);
    trace(mm)
    trace("-------------------------------- relu done -------------------------------------")
    trace()
    trace("------------------------------- conv2 start ------------------------------------")
    trace("[custom] conv2 mm: ")
    mm = custom.mm(edgeMatrix, mm)
    mm = custom.mm(mm, weight2)
    mm = custom.add(mm, bias2)
    trace(mm)
    trace("-------------------------------- conv2 done ------------------------------------")
    print("*** %s seconds ***" % round(time.time() - start_time, 2))
    trace()
  
    trace()
    print("============================== GCN geometric ===================================")
    trace()
    model.eval()
    result = model(data)
    _, pred = result.max(dim=1)

    # verify the customized output with the golden model
    if verify(mm, result):
      trace("[offload] success!")
    else:
      trace("[offload] fail!")

    correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
  
    trace('[gold] test GCN and accuracy is: {:.4f}'.format(acc))

main()
