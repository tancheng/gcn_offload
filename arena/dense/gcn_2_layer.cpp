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

//#define DUMMY_DATA

#define KERNEL_LAYER0 2
#define KERNEL_LAYER1 3

//#define num_vertice 2708
//#define num_feature 1433
//#define num_w0_out 16
//#define num_w1_out 7

//#define num_vertice 400
//#define num_feature 300
//#define num_feature 200
//#define num_w0_out 16
//#define num_w1_out 7

//#define NODES 2
#define NODES 4
//#define NODES 8
//#define NODES 16
// ----------------------------------------------------------------------
// Total data allocated onto nodes.
// TODO: user specified.
// ----------------------------------------------------------------------
//int GRAPH[num_vertice][num_vertice];

// ----------------------------------------------------------------------
// Local data allocated onto a node.
// TODO: user specified.
// ----------------------------------------------------------------------
#define SPARSE 3
int num_vertice;
int num_feature;
int num_w0_out;
int num_w1_out;
int** global_A;
float** local_A;
float** global_X;
float** local_X;
float** global_weight0;
float* global_bias0;
float** global_weight1;
float* global_bias1;
float* buff_X;
float** trans_X;
float** out0;
float** out1;
float** trans_out1;
float** out2;
float** out3;
float** temp_out;
float** output_gold;
int* offset0;
int* offset1;
bool* first_round_store;
int LAYER0_OPT_ALL;
int LAYER1_OPT_ALL;
void display_input();  // helper function
void display_output(); // helper function
bool verify(float**, float**);         // helper function

// ----------------------------------------------------------------------
// local data initialization.
// TODO: user specified.
// ----------------------------------------------------------------------
void init_data() {
#ifdef DUMMY_DATA
  num_vertice = 20;
  num_feature = 10;
  num_w0_out = 4;
  num_w1_out = 2;

  global_A = new int*[num_vertice];
  global_X = new float*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    global_A[i] = new int[num_vertice];
    for(int j=0; j<num_vertice; ++j) {
      global_A[i][j] = 0;//(i+j)%2;//(i*num_vertice+j);
    }
  }
  for(int i=0; i<num_vertice; ++i) {
    for(int j=i; j<num_vertice; ++j) {
      global_A[i][j] = j%2;
      global_A[j][i] = j%2;
    }
  }

  global_X = new float*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    global_X[i] = new float[num_feature];
    for(int j=0; j<num_feature; ++j) {
      global_X[i][j] = i*num_feature + j;//(i+j)%2;
//      if((i+j)%2 == 0)
//        global_X[i][j] = 1.0-(i*num_feature+j)%10;//(i*num_vertice+j);
//      else
//        global_X[i][j] = 1.0+(i*num_feature+j)%10;//(i*num_vertice+j);
    }
  }

  global_weight0 = new float*[num_feature];
  for(int i=0; i<num_feature; ++i) {
    global_weight0[i] = new float[num_w0_out];
    for(int j=0; j<num_w0_out; ++j) {
      global_weight0[i][j] = i;//i%3+j%3;///(num_feature*1.0);
    }
  }

  global_bias0 = new float[num_w0_out];
  for(int i=0; i<num_w0_out; ++i) {
    global_bias0[i] = 0;//i%2;///(num_feature*1.0);
  }

  global_weight1 = new float*[num_w0_out];
  for(int i=0; i<num_w0_out; ++i) {
    global_weight1[i] = new float[num_w1_out];
    for(int j=0; j<num_w1_out; ++j) {
      global_weight1[i][j] = i*num_w1_out+j;//i%3+j%3;
    }
  }

  global_bias1 = new float[num_w1_out];
  for(int i=0; i<num_w1_out; ++i) {
    global_bias1[i] = 0;//(i+1)%2;///(num_w1_out*1.0);
  }

  output_gold = new float*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    output_gold[i] = new float[num_w1_out];
  }
  output_gold[0][0] = 338;
  output_gold[1][0] = 462;
  output_gold[2][0] = 200;
  output_gold[3][0] = 548;

#endif
#ifndef DUMMY_DATA
  // read data from files and initialize input data arrays
  ifstream File;
  File.open("../../data/cora_2layer/input_a");
  File >> num_vertice;

  global_A = new int*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    global_A[i] = new int[num_vertice];
    for(int j=0; j<num_vertice; ++j) {
      File >> global_A[i][j];
    }
  }
  File.close();

  File.open("../../data/cora_2layer/input_x");
  File >> num_feature;

  global_X = new float*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    global_X[i] = new float[num_feature];
    for(int j=0; j<num_feature; ++j) {
      File >> global_X[i][j];
    }
  }
  File.close();

  File.open("../../data/cora_2layer/input_w0");
  File >> num_w0_out;

  global_weight0 = new float*[num_feature];
  for(int i=0; i<num_feature; ++i) {
    global_weight0[i] = new float[num_w0_out];
    for(int j=0; j<num_w0_out; ++j) {
      File >> global_weight0[i][j];
    }
  }

  global_bias0 = new float[num_w0_out];
  for(int i=0; i<num_w0_out; ++i) {
    File >> global_bias0[i];
  }
  File.close();

  File.open("../../data/cora_2layer/input_w1");
  File >> num_w1_out;

  global_weight1 = new float*[num_w0_out];
  for(int i=0; i<num_w0_out; ++i) {
    global_weight1[i] = new float[num_w1_out];
    for(int j=0; j<num_w1_out; ++j) {
      File >> global_weight1[i][j];
    }
  }

  global_bias1 = new float[num_w1_out];
  for(int i=0; i<num_w1_out; ++i) {
    File >> global_bias1[i];
  }
  File.close();

  File.open("../../data/cora_2layer/output_gold");

  output_gold = new float*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    output_gold[i] = new float[num_w1_out];
    for(int j=0; j<num_w1_out; ++j) {
      File >> output_gold[i][j];
    }
  }
  File.close();
#endif

  LAYER0_OPT_ALL = NODES*num_feature;
  LAYER1_OPT_ALL = NODES*num_w0_out;


  local_A = new float*[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    local_A[i] = new float[num_vertice];
    for(int j=0; j<num_vertice; ++j) {
      local_A[i][j] = global_A[ARENA_local_rank*(num_vertice/NODES)+i][j];
    }
  }

  local_X = new float*[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    local_X[i] = new float[num_feature];
    for(int j=0; j<num_feature; ++j) {
      local_X[i][j] = global_X[ARENA_local_rank*(num_vertice/NODES)+i][j];
    }
  }

  buff_X = new float[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    buff_X[i] = local_X[i][0];
  }

  trans_X = new float*[num_feature];
  for(int i=0; i<num_feature; ++i) {
    trans_X[i] = new float[num_vertice/NODES];
    for(int j=0; j<num_vertice/NODES; ++j) {
      trans_X[i][j] = local_X[j][i];
    }
  }

  trans_out1 = new float*[num_w0_out];
  for(int i=0; i<num_w0_out; ++i) {
    trans_out1[i] = new float[num_vertice/NODES];
    for(int j=0; j<num_vertice/NODES; ++j) {
      trans_out1[i][j] = 0;
    }
  }

  out0 = new float*[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    out0[i] = new float[num_feature];
    for(int j=0; j<num_feature; ++j) {
      out0[i][j] = 0;
    }
  }
  temp_out = new float*[num_feature];
  for(int i=0; i<num_feature; ++i) {
    temp_out[i] = new float[num_vertice/NODES];
    for(int j=0; j<num_vertice/NODES; ++j) {
//      temp_out[j] = 0;
      temp_out[i][j] = 0;
    }
  }

  out1 = new float*[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    out1[i] = new float[num_w0_out];
    for(int j=0; j<num_w0_out; ++j) {
      out1[i][j] = 0;
    }
  }

  out2 = new float*[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    out2[i] = new float[num_w0_out];
    for(int j=0; j<num_w0_out; ++j) {
      out2[i][j] = 0;
    }
  }

  out3 = new float*[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    out3[i] = new float[num_w1_out];
    for(int j=0; j<num_w1_out; ++j) {
      out3[i][j] = 0;
    }
  }

  // offset indicates the offset of the current region of the global nodes
  offset0 = new int[num_feature];
  for(int i=0; i<num_feature; ++i) {
    offset0[i] = 0;
  }

  offset1 = new int[num_w0_out];
  for(int i=0; i<num_w0_out; ++i) {
    offset1[i] = 0;
  }

  first_round_store= new bool[num_feature];
  for(int i=0; i<num_feature; ++i) {
    first_round_store[i] = false;
  }

}

void mw0_kernel() {
  float temp = 0;
  for(int k=0; k<num_w0_out; ++k) {
    for(int i=0; i<num_vertice/NODES; ++i) {
      temp = out1[i][k];
      for(int j=0; j<num_feature; ++j) {
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
  float temp = 0;
  for(int k=0; k<num_w1_out; ++k) {
    for(int i=0; i<num_vertice/NODES; ++i) {
      temp = out3[i][k];
      for(int j=0; j<num_w0_out; ++j) {
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
  float* current_feature = buff_X;
  k_dim = param;
  if(offset0[k_dim] == 0) {
    first_exe = true;
    first_round_store[k_dim] = true;
    current_feature = trans_X[k_dim];
  }
  offset0[k_dim] += ARENA_local_start;
  int temp_offset = offset0[k_dim];

  for(int i=0; i<num_vertice/NODES; ++i) {
    float temp = out0[i][k_dim];
    for(int j=0; j<num_vertice/NODES; ++j) {
//      if(ARENA_local_rank == NODES-1) {
//        cout<<"..[check] temp: "<<temp<<"; local: "<<local_A[i][offset0[k_dim]+j]<<"; current_feature["<<j<<"]: "<<current_feature[j]<<endl;
//      }
      temp += local_A[i][offset0[k_dim]+j] * current_feature[j];
//local_X[j][k_dim];//local_X[j][k_dim];

    }
    out0[i][k_dim] = temp;
//    if(ARENA_local_rank == NODES-1)
//      cout<<"out0["<<i<<"]["<<k_dim<<"]: "<<out0[i][k_dim]<<endl;
  }

  offset0[k_dim] -= num_vertice/NODES + ARENA_local_start;
  if(ARENA_local_start + offset0[k_dim] < 0)
    offset0[k_dim] += num_vertice;

  int num_spawn = 0;
  if(offset0[k_dim] != 0) {
    ARENA_spawn_task(KERNEL_LAYER0, ARENA_local_end%num_vertice,
                     ARENA_local_end%num_vertice+num_vertice/NODES,
                     k_dim, ARENA_local_rank, 0, num_vertice/NODES);
//    ARENA_spawn[num_spawn].id    = KERNEL_LAYER0;
//    ARENA_spawn[num_spawn].start = ARENA_local_end%num_vertice;
//    ARENA_spawn[num_spawn].end   = ARENA_local_end%num_vertice+num_vertice/NODES;
//    ARENA_spawn[num_spawn].param = k_dim;
//    ARENA_spawn[num_spawn].more_from  = ARENA_local_rank;
//    ARENA_spawn[num_spawn].more_start = 0;
//    ARENA_spawn[num_spawn].more_end   = num_vertice/NODES;
    ARENA_remote_ask_start[(ARENA_local_rank+1)%NODES] = 0;
    ARENA_remote_ask_end[(ARENA_local_rank+1)%NODES] = num_vertice/NODES;
    ++num_spawn;

    if(first_exe and k_dim < num_feature-1) {
      ARENA_spawn_task(KERNEL_LAYER0, ARENA_local_start,
                       ARENA_local_end, k_dim+1);
      ++num_spawn;
    }
  }

  if(opt_count == LAYER0_OPT_ALL) {
    mw0_kernel();
    opt_count = 0;
    k_dim = 0;
    ARENA_spawn_task(KERNEL_LAYER1, ARENA_local_start,
                     ARENA_local_end, k_dim);
    ++num_spawn;
  }

  return -1;//num_spawn;
}

int ARENA_kernel1(int start, int end, int param) {
  layer = 1;
  ++opt_count;
  bool first_exe = false;
  float* current_feature = buff_X;
  k_dim = param;
  if(offset0[k_dim] == 0) {
    first_exe = true;
    first_round_store[k_dim] = true;
    current_feature = trans_out1[k_dim];
  }
  offset0[k_dim] += ARENA_local_start;

  for(int i=0; i<num_vertice/NODES; ++i) {
    float temp = out2[i][k_dim];
    for(int j=0; j<num_vertice/NODES; ++j) {
      temp += local_A[i][offset0[k_dim]+j] * current_feature[j];
//local_X[j][k_dim];//local_X[j][k_dim];
//      if(ARENA_local_rank == 3)
//        cout<<"..[check] temp: "<<temp<<"; current_feature["<<j<<"]: "<<current_feature[j]<<endl;
    }
    out2[i][k_dim] = temp;
//    if(ARENA_local_rank == 3)
//      cout<<"..[check] out2["<<i<<"]["<<k_dim<<"]: "<<out2[i][k_dim]<<endl;
  }

  offset0[k_dim] -= num_vertice/NODES + ARENA_local_start;
  if(ARENA_local_start + offset0[k_dim] < 0)
    offset0[k_dim] += num_vertice;

  int num_spawn = 0;
  if(offset0[k_dim] != 0) {
    ARENA_spawn_task(KERNEL_LAYER1, ARENA_local_end%num_vertice,
                     ARENA_local_end%num_vertice+num_vertice/NODES,
                     k_dim, ARENA_local_rank, 0, num_vertice/NODES);
    ARENA_remote_ask_start[(ARENA_local_rank+1)%NODES] = 0;
    ARENA_remote_ask_end[(ARENA_local_rank+1)%NODES] = num_vertice/NODES;
    ++num_spawn;

    if(first_exe and k_dim < num_w0_out-1) {
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

  init_data();

  // TODO: Data tag.
  ARENA_local_bound = rank * (num_vertice/NODES);
  ARENA_local_start = rank * (num_vertice/NODES);
  ARENA_local_end   = rank * (num_vertice/NODES) + (num_vertice/NODES);

  // TODO: Task start point.
  ARENA_global_start = 0;//ARENA_local_start;
  ARENA_global_end   = num_vertice;//ARENA_local_end;
  ARENA_global_param = 0;


  // TODO: Remote data requirement. The second parameter indicates
  //       wheter the data depends on the predecessor task
  ARENA_init_data_buff(num_vertice/NODES, false);
  ARENA_remote_ask_buff[(rank+1)%NODES] = new float[num_vertice/NODES];
  ARENA_local_need_buff[(num_vertice+rank-1)%NODES] = new float[num_vertice/NODES];
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
  // display_input();

  // Execute kernel
  ARENA_run();

  // Verify
  if(verify(out3, output_gold)) {
    cout<<"rank "<<ARENA_local_rank<<" success~"<<endl;
  } else {
    cout<<"rank "<<ARENA_local_rank<<" fail.."<<endl;
//    display_output();
  }

  // Output
  if(ARENA_local_rank == NODES-1) {
    cout<<"[final] rank "<<ARENA_local_rank<<" out0: "<<endl;
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_feature-1; j<num_feature; ++j) {
//    for(int i=0; i<num_vertice/NODES; ++i) {
//      cout<<"[ ";
//      for(int j=0; j<num_feature; ++j) {
        cout<<"out0["<<i<<"]["<<j<<"]: "<<out0[i][j]<<"; ";
      }
      cout<<" ]"<<endl;
    }
    cout<<"[final] rank "<<ARENA_local_rank<<" out1: "<<endl;
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_w0_out-1; j<num_w0_out; ++j) {
//    for(int i=0; i<num_vertice/NODES; ++i) {
//      cout<<"[ ";
//      for(int j=0; j<num_w0_out; ++j) {
        cout<<out1[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }

    cout<<"[final] rank "<<ARENA_local_rank<<" trans_out1: "<<endl;
    for(int i=num_w0_out-1; i<num_w0_out; ++i) {
      cout<<"[ ";
      for(int j=num_vertice/NODES-1; j<num_vertice/NODES; ++j) {
//    for(int i=0; i<num_w0_out; ++i) {
//      cout<<"[ ";
//      for(int j=0; j<num_vertice/NODES; ++j) {
        cout<<trans_out1[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }

    cout<<"[final] rank "<<ARENA_local_rank<<" out2: "<<endl;
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_w0_out-1; j<num_w0_out; ++j) {
//    for(int i=0; i<num_vertice/NODES; ++i) {
//      cout<<"[ ";
//      for(int j=0; j<num_w0_out; ++j) {
        cout<<out2[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }

    cout<<"[final] rank "<<ARENA_local_rank<<" out3: "<<endl;
//    for(int i=0; i<num_vertice/NODES; ++i) {
//      cout<<"[ ";
//      for(int j=0; j<num_w1_out; ++j) {
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=0; j<num_w1_out; ++j) {
        cout<<out3[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
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


// ======================================================================
// helper functions (e.g., print out input/output, verify results)
// ----------------------------------------------------------------------
void display_input() {
  if (ARENA_local_rank == NODES-1) { 
  cout<<"[init A] rank "<<ARENA_local_rank<<" : "<<num_vertice/NODES<<": "<<endl;
  for(int i=0; i<num_vertice/NODES; ++i) {
    cout<<"[ ";
    for(int j=0; j<num_vertice; ++j) {
      cout<<local_A[i][j]<<" ";
    }
    cout<<" ]"<<endl;
  }
  cout<<"[init X] rank "<<ARENA_local_rank<<" : "<<endl;
  for(int i=0; i<num_vertice/NODES; ++i) {
    cout<<"[ ";
    for(int j=0; j<num_feature; ++j) {
      cout<<local_X[i][j]<<" ";
    }
    cout<<" ]"<<endl;
  }
  cout<<"[init weight1] rank "<<ARENA_local_rank<<" : "<<endl;
  for(int i=0; i<num_feature; ++i) {
    cout<<"[ ";
    for(int j=0; j<num_w0_out; ++j) {
      cout<<global_weight0[i][j]<<" ";
    }
    cout<<" ]"<<endl;
  }

  cout<<"[init bias1] rank "<<ARENA_local_rank<<" : "<<endl;
  cout<<"[ ";
  for(int j=0; j<num_w0_out; ++j) {
    cout<<global_bias0[j]<<" ";
  }
  cout<<" ]"<<endl;

  cout<<"[init weight2] rank "<<ARENA_local_rank<<" : "<<endl;
  for(int i=0; i<num_w0_out; ++i) {
    cout<<"[ ";
    for(int j=0; j<num_w1_out; ++j) {
      cout<<global_weight1[i][j]<<" ";
    }
    cout<<" ]"<<endl;
  }

  cout<<"[init bias2] rank "<<ARENA_local_rank<<" : "<<endl;
  cout<<"[ ";
  for(int j=0; j<num_w1_out; ++j) {
    cout<<global_bias1[j]<<" ";
  }
  cout<<" ]"<<endl;
  }
}

void display_output() {
//  if(local_rank == NODES-1) {
    cout<<"[output out0] rank "<<ARENA_local_rank<<" : "<<endl;
    for(int i=0; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=0; j<num_feature; ++j) {
        cout<<out0[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
    cout<<"[output out1] rank "<<ARENA_local_rank<<" : "<<endl;
    for(int i=0; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=0; j<num_w0_out; ++j) {
        cout<<out1[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
//  }
}

bool verify(float** a, float** b) {
  int base = ARENA_local_rank*(num_vertice/NODES);
  for(int i=ARENA_local_rank*(num_vertice/NODES); i<(ARENA_local_rank+1)*(num_vertice/NODES); ++i) {
    for(int j=0; j<num_w1_out; ++j) {
      if(abs(a[i-base][j] - b[i][j]) > 0.001) {
        cout<<"rank: "<<ARENA_local_rank<<" local_out["<<i-base<<"]["<<j<<"]: "<<a[i-base][j]<<"; gold["<<i<<"]["<<j<<"]: "<<b[i][j]<<endl;
        return false;
      }
    }
  }
  return true;
}
