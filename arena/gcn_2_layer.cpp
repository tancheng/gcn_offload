// =======================================================================
// gcn_ax.cpp
// =======================================================================
// ARENA implementation of single layer GCN's ax.
// Note that the entire computation of single layer GCN is FC(gemv(A,x), W).
//
// Mechanism: Sparse matrix multiplication on its local data then stream
//            the data to the next location indicated by start/end.
//            Theoratically, it has the same amount of data movement as
//            the conventional bulk-synchronization MPI (broadcast-based)
//            solution.
//
// Benefit:   The computaton and communication can be asynchronous compared
//            to the bulk-synchronization solution.
//
// Problem:   Tried to think about it in a push-based style, but the task
//            delivery would probably dominate the communication (i.e.,
//            data transfer are translated into task transfer in the
//            context of ARENA). So still make it pull-based (then stream
//            the local features).
//
// TODO: 1. Vector -> Matrix.
//       2. Load CORA 2-layer weights.
//       3. Invoke second layer workload by spawning new tasks.
//
// Author : Cheng Tan
//   Date : June 15, 2020

#include "../lib/ARENA.h"
#include <iostream>
#include <fstream>
#include <string>

#define KERNEL_LAYER0 2
#define KERNEL_LAYER1 3

//#define SIZE_IN 2708
//#define SIZE_FEATURE 1433
//#define SIZE_OUT1 16
//#define SIZE_OUT2 7

#define SIZE_IN 400
#define SIZE_FEATURE 300
//#define SIZE_FEATURE 200
#define SIZE_OUT1 16
#define SIZE_OUT2 7

//#define NODES 2
#define NODES 4
//#define NODES 8
//#define NODES 16
// ----------------------------------------------------------------------
// Total data allocated onto nodes.
// TODO: user specified.
// ----------------------------------------------------------------------
//int GRAPH[SIZE_IN][SIZE_IN];

// ----------------------------------------------------------------------
// Local data allocated onto a node.
// TODO: user specified.
// ----------------------------------------------------------------------
#define SPARSE 3
int** global_A;
int** local_A;
int** global_X;
int** local_X;
int** global_weight0;
int* global_bias0;
int** global_weight1;
int* global_bias1;
int* buff_X;
int** trans_X;
int** out0;
int** out1;
int** trans_out1;
int** out2;
int** out3;
int** temp_out;
int* offset0;
int* offset1;
bool* first_round_store;
int LAYER0_OPT_ALL;
int LAYER1_OPT_ALL;

void init_data() {

  LAYER0_OPT_ALL = NODES*SIZE_FEATURE;
  LAYER1_OPT_ALL = NODES*SIZE_OUT1;

  global_A = new int*[SIZE_IN];
  global_X = new int*[SIZE_IN];
  for(int i=0; i<SIZE_IN; ++i) {
    global_A[i] = new int[SIZE_IN];
    for(int j=0; j<SIZE_IN; ++j) {
      global_A[i][j] = 1;//(i+j)%2;//(i*SIZE_IN+j);
    }

    global_X[i] = new int[SIZE_FEATURE];
    for(int j=0; j<SIZE_FEATURE; ++j) {
      global_X[i][j] = i*SIZE_FEATURE+j;//(i+j)%2;//(i*SIZE_FEATURE + j)%10;
//      if((i+j)%2 == 0)
//        global_X[i][j] = 1.0-(i*SIZE_FEATURE+j)%10;//(i*SIZE_IN+j);
//      else
//        global_X[i][j] = 1.0+(i*SIZE_FEATURE+j)%10;//(i*SIZE_IN+j);
    }
  }


  local_A = new int*[SIZE_IN/NODES];
  for(int i=0; i<SIZE_IN/NODES; ++i) {
    local_A[i] = new int[SIZE_IN];
    for(int j=0; j<SIZE_IN; ++j) {
      local_A[i][j] = global_A[ARENA_local_rank*(SIZE_IN/NODES)+i][j];
    }
  }

  local_X = new int*[SIZE_IN/NODES];
  for(int i=0; i<SIZE_IN/NODES; ++i) {
    local_X[i] = new int[SIZE_FEATURE];
    for(int j=0; j<SIZE_FEATURE; ++j) {
      local_X[i][j] = global_X[ARENA_local_rank*(SIZE_IN/NODES)+i][j];
    }
  }

  buff_X = new int[SIZE_IN/NODES];
  for(int i=0; i<SIZE_IN/NODES; ++i) {
    buff_X[i] = local_X[i][0];
  }

  trans_X = new int*[SIZE_FEATURE];
  for(int i=0; i<SIZE_FEATURE; ++i) {
    trans_X[i] = new int[SIZE_IN/NODES];
    for(int j=0; j<SIZE_IN/NODES; ++j) {
      trans_X[i][j] = local_X[j][i];
    }
  }

  trans_out1 = new int*[SIZE_OUT1];
  for(int i=0; i<SIZE_OUT1; ++i) {
    trans_out1[i] = new int[SIZE_IN/NODES];
    for(int j=0; j<SIZE_IN/NODES; ++j) {
      trans_out1[i][j] = 0;
    }
  }

  global_weight0 = new int*[SIZE_FEATURE];
  for(int i=0; i<SIZE_FEATURE; ++i) {
    global_weight0[i] = new int[SIZE_OUT1];
    for(int j=0; j<SIZE_OUT1; ++j) {
      global_weight0[i][j] = i%3+j%3;///(SIZE_FEATURE*1.0);
    }
  }

  global_bias0 = new int[SIZE_OUT1];
  for(int i=0; i<SIZE_OUT1; ++i) {
    global_bias0[i] = i%2;///(SIZE_FEATURE*1.0);
  }

  global_weight1 = new int*[SIZE_OUT1];
  for(int i=0; i<SIZE_OUT1; ++i) {
    global_weight1[i] = new int[SIZE_OUT2];
    for(int j=0; j<SIZE_OUT2; ++j) {
      global_weight1[i][j] = i%3+j%3;///(SIZE_OUT1*1.0);
    }
  }

  global_bias1 = new int[SIZE_OUT2];
  for(int i=0; i<SIZE_OUT2; ++i) {
    global_bias1[i] = (i+1)%2;///(SIZE_OUT2*1.0);
  }

  out0 = new int*[SIZE_IN/NODES];
  for(int i=0; i<SIZE_IN/NODES; ++i) {
    out0[i] = new int[SIZE_FEATURE];
    for(int j=0; j<SIZE_FEATURE; ++j) {
      out0[i][j] = 0;
    }
  }
  temp_out = new int*[SIZE_FEATURE];
  for(int i=0; i<SIZE_FEATURE; ++i) {
    temp_out[i] = new int[SIZE_IN/NODES];
    for(int j=0; j<SIZE_IN/NODES; ++j) {
//      temp_out[j] = 0;
      temp_out[i][j] = 0;
    }
  }

  out1 = new int*[SIZE_IN/NODES];
  for(int i=0; i<SIZE_IN/NODES; ++i) {
    out1[i] = new int[SIZE_OUT1];
    for(int j=0; j<SIZE_OUT1; ++j) {
      out1[i][j] = 0;
    }
  }

  out2 = new int*[SIZE_IN/NODES];
  for(int i=0; i<SIZE_IN/NODES; ++i) {
    out2[i] = new int[SIZE_OUT1];
    for(int j=0; j<SIZE_OUT1; ++j) {
      out2[i][j] = 0;
    }
  }

  out3 = new int*[SIZE_IN/NODES];
  for(int i=0; i<SIZE_IN/NODES; ++i) {
    out3[i] = new int[SIZE_OUT2];
    for(int j=0; j<SIZE_OUT2; ++j) {
      out3[i][j] = 0;
    }
  }

  // offset indicates the offset of the current region of the global nodes
  offset0 = new int[SIZE_FEATURE];
  for(int i=0; i<SIZE_FEATURE; ++i) {
    offset0[i] = 0;
  }

  offset1 = new int[SIZE_OUT1];
  for(int i=0; i<SIZE_OUT1; ++i) {
    offset1[i] = 0;
  }

  first_round_store= new bool[SIZE_FEATURE];
  for(int i=0; i<SIZE_FEATURE; ++i) {
    first_round_store[i] = false;
  }

}

void display() {
  if (ARENA_local_rank == 0) {
  cout<<"[init A] rank "<<ARENA_local_rank<<" : "<<SIZE_IN/NODES<<": "<<endl;
  for(int i=0; i<SIZE_IN/NODES; ++i) {
    cout<<"[ ";
    for(int j=0; j<SIZE_IN; ++j) {
      cout<<local_A[i][j]<<" ";
    }
    cout<<" ]"<<endl;
  }
  cout<<"[init X] rank "<<ARENA_local_rank<<" : "<<endl;
  for(int i=0; i<SIZE_IN/NODES; ++i) {
    cout<<"[ ";
    for(int j=0; j<SIZE_FEATURE; ++j) {
      cout<<local_X[i][j]<<" ";
    }
    cout<<" ]"<<endl;
  }
  cout<<"[init weight1] rank "<<ARENA_local_rank<<" : "<<endl;
  for(int i=0; i<SIZE_FEATURE; ++i) {
    cout<<"[ ";
    for(int j=0; j<SIZE_OUT1; ++j) {
      cout<<global_weight0[i][j]<<" ";
    }
    cout<<" ]"<<endl;
  }

  cout<<"[init bias1] rank "<<ARENA_local_rank<<" : "<<endl;
  cout<<"[ ";
  for(int j=0; j<SIZE_OUT1; ++j) {
    cout<<global_bias0[j]<<" ";
  }
  cout<<" ]"<<endl;

  cout<<"[init weight2] rank "<<ARENA_local_rank<<" : "<<endl;
  for(int i=0; i<SIZE_OUT1; ++i) {
    cout<<"[ ";
    for(int j=0; j<SIZE_OUT2; ++j) {
      cout<<global_weight1[i][j]<<" ";
    }
    cout<<" ]"<<endl;
  }

  cout<<"[init bias2] rank "<<ARENA_local_rank<<" : "<<endl;
  cout<<"[ ";
  for(int j=0; j<SIZE_OUT2; ++j) {
    cout<<global_bias1[j]<<" ";
  }
  cout<<" ]"<<endl;
  }
}

void mw0_kernel() {
  int temp = 0;
  for(int k=0; k<SIZE_OUT1; ++k) {
    for(int i=0; i<SIZE_IN/NODES; ++i) {
      temp = out1[i][k];
      for(int j=0; j<SIZE_FEATURE; ++j) {
        temp += out0[i][j] * global_weight0[j][k];
      }
      out1[i][k] = temp + global_bias0[k];
      if(out1[i][k] < 0)
        out1[i][k] = 0;
      trans_out1[k][i] = out1[i][k];
    }
  }
}

void mw1_kernel() {
  int temp = 0;
  for(int k=0; k<SIZE_OUT2; ++k) {
    for(int i=0; i<SIZE_IN/NODES; ++i) {
      temp = out3[i][k];
      for(int j=0; j<SIZE_OUT1; ++j) {
        temp += out2[i][j] * global_weight1[j][k];
      }
      out3[i][k] = temp + global_bias1[k];
    }
  }
}

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc).
// Note that there are three params and one return.
// TODO: user specified.
// ----------------------------------------------------------------------
//int total_times = 0;
int layer = 0;
int opt_count = 0;
int k_dim = 0;
int ARENA_kernel0(int start, int end, int param) {
  layer = 0;
  ++opt_count;
  bool first_exe = false;
  int* current_feature = buff_X;
  k_dim = param;
  if(offset0[k_dim] == 0) {
    first_exe = true;
    first_round_store[k_dim] = true;
    current_feature = trans_X[k_dim];
  }
  offset0[k_dim] += ARENA_local_start;
  int temp_offset = offset0[k_dim];

  for(int i=0; i<SIZE_IN/NODES; ++i) {
    int temp = out0[i][k_dim];
    for(int j=0; j<SIZE_IN/NODES; ++j) {
//      if(ARENA_local_rank == 0) {
//        cout<<"..[check] temp: "<<temp<<"; local: "<<local_A[i][offset0[k_dim]+j]<<"; current_feature["<<j<<"]: "<<current_feature[j]<<endl;
//      }
      temp += local_A[i][offset0[k_dim]+j] * current_feature[j];
//local_X[j][k_dim];//local_X[j][k_dim];

    }
    out0[i][k_dim] = temp;
//    if(ARENA_local_rank == 3)
//      cout<<"out0["<<i<<"]["<<k_dim<<"]: "<<out0[i][k_dim]<<endl;
  }

  offset0[k_dim] -= SIZE_IN/NODES + ARENA_local_start;
  if(ARENA_local_start + offset0[k_dim] < 0)
    offset0[k_dim] += SIZE_IN;

  int num_spawn = 0;
  if(offset0[k_dim] != 0) {
    ARENA_spawn_task(KERNEL_LAYER0, ARENA_local_end%SIZE_IN,
                     ARENA_local_end%SIZE_IN+SIZE_IN/NODES,
                     k_dim, ARENA_local_rank, 0, SIZE_IN/NODES);
//    ARENA_spawn[num_spawn].id    = KERNEL_LAYER0;
//    ARENA_spawn[num_spawn].start = ARENA_local_end%SIZE_IN;
//    ARENA_spawn[num_spawn].end   = ARENA_local_end%SIZE_IN+SIZE_IN/NODES;
//    ARENA_spawn[num_spawn].param = k_dim;
//    ARENA_spawn[num_spawn].more_from  = ARENA_local_rank;
//    ARENA_spawn[num_spawn].more_start = 0;
//    ARENA_spawn[num_spawn].more_end   = SIZE_IN/NODES;
    ARENA_remote_ask_start[(ARENA_local_rank+1)%NODES] = 0;
    ARENA_remote_ask_end[(ARENA_local_rank+1)%NODES] = SIZE_IN/NODES;
    ++num_spawn;

    if(first_exe and k_dim < SIZE_FEATURE-1) {
      ARENA_spawn_task(KERNEL_LAYER0, ARENA_local_start,
                       ARENA_local_end, k_dim+1);
      ++num_spawn;
    }
  }

  if(opt_count == LAYER0_OPT_ALL) {
    mw0_kernel();
    opt_count = 0;
    k_dim = 0;
//    ARENA_spawn_task(KERNEL_LAYER1, ARENA_local_start,
//                     ARENA_local_end, k_dim);
//    ++num_spawn;
  }

  return -1;//num_spawn;
}

int ARENA_kernel1(int start, int end, int param) {
  layer = 1;
  ++opt_count;
  bool first_exe = false;
  int* current_feature = buff_X;
  k_dim = param;
  if(offset0[k_dim] == 0) {
    first_exe = true;
    first_round_store[k_dim] = true;
    current_feature = trans_out1[k_dim];
  }
  offset0[k_dim] += ARENA_local_start;

  for(int i=0; i<SIZE_IN/NODES; ++i) {
    int temp = out2[i][k_dim];
    for(int j=0; j<SIZE_IN/NODES; ++j) {
      temp += local_A[i][offset0[k_dim]+j] * current_feature[j];
//local_X[j][k_dim];//local_X[j][k_dim];
//      if(ARENA_local_rank == 3)
//        cout<<"..[check] temp: "<<temp<<"; current_feature["<<j<<"]: "<<current_feature[j]<<endl;
    }
    out2[i][k_dim] = temp;
//    if(ARENA_local_rank == 3)
//      cout<<"..[check] out2["<<i<<"]["<<k_dim<<"]: "<<out2[i][k_dim]<<endl;
  }

  offset0[k_dim] -= SIZE_IN/NODES + ARENA_local_start;
  if(ARENA_local_start + offset0[k_dim] < 0)
    offset0[k_dim] += SIZE_IN;

  int num_spawn = 0;
  if(offset0[k_dim] != 0) {
    ARENA_spawn_task(KERNEL_LAYER1, ARENA_local_end%SIZE_IN,
                     ARENA_local_end%SIZE_IN+SIZE_IN/NODES,
                     k_dim, ARENA_local_rank, 0, SIZE_IN/NODES);
    ARENA_remote_ask_start[(ARENA_local_rank+1)%NODES] = 0;
    ARENA_remote_ask_end[(ARENA_local_rank+1)%NODES] = SIZE_IN/NODES;
    ++num_spawn;

    if(first_exe and k_dim < SIZE_OUT1-1) {
      ARENA_spawn_task(KERNEL_LAYER1, ARENA_local_start,
                       ARENA_local_end, k_dim+1);
      ++num_spawn;
    }
  }

  if(opt_count == LAYER1_OPT_ALL) {
    mw1_kernel();
  }

  return -1;
}

void ARENA_init(int argc, char *argv[], int nodes) {
  // MPI initial
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &ARENA_nodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //ARENA_nodes = nodes;
  ARENA_local_rank = rank;

  // TODO: Data tag.
  ARENA_local_bound = rank * (SIZE_IN/NODES);
  ARENA_local_start = rank * (SIZE_IN/NODES);
  ARENA_local_end   = rank * (SIZE_IN/NODES) + (SIZE_IN/NODES);

  // TODO: Task start point.
  ARENA_global_start = 0;//ARENA_local_start;
  ARENA_global_end   = SIZE_IN;//ARENA_local_end;
  ARENA_global_param = 0;

  init_data();

  // TODO: Remote data requirement. The second parameter indicates
  //       wheter the data depends on the predecessor task
  ARENA_init_data_buff(SIZE_IN/NODES, false);
  ARENA_remote_ask_buff[(rank+1)%NODES] = new float[SIZE_IN/NODES];
  ARENA_local_need_buff[(SIZE_IN+rank-1)%NODES] = new float[SIZE_IN/NODES];
}

// ----------------------------------------------------------------------
// Main function. No need to change.
// ----------------------------------------------------------------------
int main(int argc, char *argv[]) {

  // Initialize global data start and end
  ARENA_init(argc, argv, NODES);

  ARENA_register(KERNEL_LAYER0, &ARENA_kernel0, true);
  ARENA_register(KERNEL_LAYER1, &ARENA_kernel1, false);

  // Display local allocated data
  // display();

  // Execute kernel
  ARENA_run();

  // Output
  if(ARENA_local_rank == 0) {
    cout<<"[final] rank "<<ARENA_local_rank<<" out0: "<<endl;
    for(int i=SIZE_IN/NODES-1; i<SIZE_IN/NODES; ++i) {
      cout<<"[ ";
      for(int j=SIZE_FEATURE-1; j<SIZE_FEATURE; ++j) {
//    for(int i=0; i<SIZE_IN/NODES; ++i) {
//      cout<<"[ ";
//      for(int j=0; j<SIZE_FEATURE; ++j) {
        cout<<"out0["<<i<<"]["<<j<<"]: "<<out0[i][j]<<"; ";
      }
      cout<<" ]"<<endl;
    }
/*
    cout<<"[final] rank "<<ARENA_local_rank<<" out1: "<<endl;
    for(int i=SIZE_IN/NODES-1; i<SIZE_IN/NODES; ++i) {
      cout<<"[ ";
      for(int j=SIZE_OUT1-1; j<SIZE_OUT1; ++j) {
//    for(int i=0; i<SIZE_IN/NODES; ++i) {
//      cout<<"[ ";
//      for(int j=0; j<SIZE_OUT1; ++j) {
        cout<<out1[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }

    cout<<"[final] rank "<<ARENA_local_rank<<" trans_out1: "<<endl;
    for(int i=SIZE_OUT1-1; i<SIZE_OUT1; ++i) {
      cout<<"[ ";
      for(int j=SIZE_IN/NODES-1; j<SIZE_IN/NODES; ++j) {
//    for(int i=0; i<SIZE_OUT1; ++i) {
//      cout<<"[ ";
//      for(int j=0; j<SIZE_IN/NODES; ++j) {
        cout<<trans_out1[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }

    cout<<"[final] rank "<<ARENA_local_rank<<" out2: "<<endl;
    for(int i=SIZE_IN/NODES-1; i<SIZE_IN/NODES; ++i) {
      cout<<"[ ";
      for(int j=SIZE_OUT1-1; j<SIZE_OUT1; ++j) {
//    for(int i=0; i<SIZE_IN/NODES; ++i) {
//      cout<<"[ ";
//      for(int j=0; j<SIZE_OUT1; ++j) {
        cout<<out2[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }

    cout<<"[final] rank "<<ARENA_local_rank<<" out3: "<<endl;
//    for(int i=0; i<SIZE_IN/NODES; ++i) {
//      cout<<"[ ";
//      for(int j=0; j<SIZE_OUT2; ++j) {
    for(int i=SIZE_IN/NODES-1; i<SIZE_IN/NODES; ++i) {
      cout<<"[ ";
      for(int j=SIZE_OUT2-1; j<SIZE_OUT2; ++j) {
        cout<<out3[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
*/
  }
  return 0;
}

// ----------------------------------------------------------------------
// Prepare data to send to remote nodes.
// TODO: user specified if necessary
// ----------------------------------------------------------------------
void ARENA_load_data(int start, int end, float* buff) {
  if(first_round_store[k_dim]) {
    first_round_store[k_dim] = false;
    if(layer == 0) {
      for(int i=start; i<end; ++i) {
        // buff[i] = local_X[i][k_dim];
        // buff[i] = buff_X[i];
          buff[i] = trans_X[k_dim][i];
      }
    } else {
       for(int i=start; i<end; ++i) {
        // buff[i] = local_X[i][k_dim];
        // buff[i] = buff_X[i];
          buff[i] = trans_out1[k_dim][i];
      }
    }
  } else {
    for(int i=start; i<end; ++i) {
      buff[i] = buff_X[i];
    }
  }
//  if(ARENA_local_rank == 3) {
//    cout<<"....[send data]: ";
//    for(int i=start; i<end; ++i) {
//      cout<<buff[i]<<" ";
//    }
//    cout<<endl;
//  }
  return;
}

// ----------------------------------------------------------------------
// Receive data from remote nodes and store into local memory.
// TODO: user specified if necessary
// ----------------------------------------------------------------------
int test = 0;
void ARENA_store_data(int start, int end, int source, float* buff) {
  for(int i=start; i<end; ++i) {
    // local_X[i][k_dim] = buff[i];
    buff_X[i] = buff[i];
    // trans_X[k_dim][i] = buff[i];
  }
//  if(ARENA_local_rank == 3) {
//    int temp = 0;
//    cout<<"....[recv data]: ";
//    for(int i=start; i<end; ++i) {
//      if(buff[i] == 1)
//        temp += 1;
//      if((i+test)%2 == buff[i]) {
//        cout<<"[mismatch...]";
//      }
//      cout<<buff[i];
//    }
//    cout<<endl;
//    if(temp != 50) {
//      cout<<"[fk!]"<<endl;
//    }
//  }
  test++;
  return;
}

