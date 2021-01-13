// ======================================================================
// gcn_2_layer.cpp
// ======================================================================
// ARENA implementation of 2 layer GCN for CORA dataset.
// The adjacent Matrix is reprensented in COO format. 
//
// Mechanism: Sparse matrix multiplication on its local data then stream
//            the data to the next location indicated by start/end.
//            Theoratically, it has the same amount of data movement as
//            the conventional bulk-synchronization MPI (broadcast-based)
//            solution.
//
// Benefit:   The computaton and communication can be asynchronous
//            compared to the bulk-synchronization solution.
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
//   Date : Jan 5, 2021

#include "../lib/ARENA.h"
#include <iostream>
#include <fstream>
#include <string>

//#define DUMMY_DATA

#define KERNEL_LAYER0 2
#define KERNEL_LAYER0_ACCUM 3
#define KERNEL_LAYER1 4
#define KERNEL_LAYER1_ACCUM 5

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
float** recv_X;
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
void display_input();          // helper function
void display_output();         // helper function
bool verify(float**, float**); // helper function

int num_nonzero_A = 0;         // for csr format
int local_nnz;                 // for csr format
float* local_V;                // for csr format
int* local_COL;                // for csr format
int* local_ROW;                // for csr format

int* data_send_times;
int* data_recv_times;

// ----------------------------------------------------------------------
// local data initialization.
// TODO: user specified.
// ----------------------------------------------------------------------
void init_data() {
#ifdef DUMMY_DATA
  num_vertice = 8;
  num_feature = 4;
  num_w0_out = 2;
  num_w1_out = 1;

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
  output_gold[0][0] = 2380;
  output_gold[1][0] = 2820;
  output_gold[2][0] = 1926;
  output_gold[3][0] = 3222;
  output_gold[4][0] = 1410;
  output_gold[5][0] = 3538;
  output_gold[6][0] = 784;
  output_gold[7][0] = 3720;

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

  data_send_times = new int[num_vertice/NODES];
  data_recv_times = new int[NODES];
  for(int i=0; i<NODES; ++i) {
    data_recv_times[i] = 0;
  }

  for(int i=0; i<num_vertice; ++i) {
    for(int j=0; j<num_vertice; ++j) {
      if(global_A[i][j] != 0) {
        num_nonzero_A += 1;
      }
    }
  }

  local_nnz = 0;
  for(int i=ARENA_local_rank*num_vertice/NODES; i<(ARENA_local_rank+1)*num_vertice/NODES; ++i) {
    for(int j=0; j<num_vertice; ++j) {
      if(global_A[i][j] != 0) {
        local_nnz += 1;
      }
    }
  }
  local_V = new float[local_nnz];
  local_COL = new int[local_nnz];
  local_ROW = new int[num_vertice/NODES+1];

  int temp = 0;
  local_ROW[0] = 0;
//  cout<<"rank "<<local_rank<<" local_ROW[0]: "<<local_ROW[0]<<endl;
  for(int i=ARENA_local_rank*num_vertice/NODES; i<(ARENA_local_rank+1)*num_vertice/NODES; ++i) {
    for(int j=0; j<num_vertice; ++j) {
      if(global_A[i][j] != 0) {
        local_V[temp] = global_A[i][j];
        local_COL[temp] = j;
//        cout<<"rank "<<local_rank<<" local_V["<<temp<<"]: "<<local_V[temp]<<endl;
//        cout<<"rank "<<local_rank<<" local_COL["<<temp<<"]: "<<local_COL[temp]<<endl;
        ++temp;
      }
    }
    local_ROW[i-ARENA_local_rank*num_vertice/NODES+1] = temp;
//    cout<<"rank "<<local_rank<<" local_ROW["<<i-local_rank*local_bound+1<<"]: "<<temp<<endl;
  }

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

  int local_start = ARENA_local_rank*(num_vertice/NODES);
  int local_end = (ARENA_local_rank+1)*(num_vertice/NODES);

  buff_X = new float[num_feature];
  recv_X = new float*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    recv_X[i] = new float[num_feature];
    for(int j=0; j<num_feature; ++j) {
      recv_X[i][j] = 0;
      if(i >= local_start and i < local_end) {
        recv_X[i][j] = local_X[i-local_start][j];
      }
    }
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
int opt_count = 0;
int range;
int ARENA_kernel0(int start, int end, int param) {
  range = num_vertice/NODES;

  data_send_times[param] = 0;
  bool* sent_tag = new bool[NODES];
  for(int i=0; i<NODES; ++i) {
    sent_tag[i] = false;
  }
  // iterate across the value inside a specific row
  for(int i=local_ROW[param]; i<local_ROW[param+1]; ++i) {
    // if the index is inside my own data range, accumulate it locally
    if(local_COL[i] >= ARENA_local_start and
       local_COL[i] < ARENA_local_end) {
      for(int x=0; x<num_feature; ++x) {
//        cout<<"rank "<<ARENA_local_rank<<" ready to self partial sum local["<<param<<"]["<<x<<"] "<<local_X[param][x]<<" add to out0["<<local_COL[i]-ARENA_local_start<<"]["<<x<<"] "<<out0[local_COL[i]-ARENA_local_start][x]<<endl;
        out0[local_COL[i]-ARENA_local_start][x] += local_X[param][x];
      }
      // count the number of operation to guide the start of next layer
      ++opt_count;
    } else {
      // NOTE that ARENA does not allow send to one destination multiple times at one shot! Thus, we only send features to a destination once.
      int dest = local_COL[i]/range;
      if(!sent_tag[dest]) {
//        cout<<"rank "<<ARENA_local_rank<<" spawn remote task at row "<<param<<" from "<<local_COL[i]<<" to "<<local_COL[i]+1<<" and send features"<<endl;
        sent_tag[dest] = true;
        ARENA_spawn_task(KERNEL_LAYER0_ACCUM, local_COL[i],
                         local_COL[i]+1, param+ARENA_local_start,
                         ARENA_local_rank, 0, num_feature);
        data_send_times[param] += 1;
      } else {
//        cout<<"rank "<<ARENA_local_rank<<" spawn remote task at row "<<param<<" from "<<local_COL[i]<<" to "<<local_COL[i]+1<<" WITHOUT send features"<<endl;
        ARENA_spawn_task(KERNEL_LAYER0_ACCUM, local_COL[i],
                         local_COL[i]+1, param+ARENA_local_start,
                         ARENA_local_rank, 0, 0);
      }
      ARENA_remote_ask_start[local_COL[i]/range] = 0;
      ARENA_remote_ask_end[local_COL[i]/range] = num_feature;

    }
  }
  
  // iterating the rows by spawning new local tasks untill the boundary
  if(param < range-1) {
    ARENA_spawn_task(KERNEL_LAYER0, ARENA_local_start, ARENA_local_end,
                     param+1, ARENA_local_rank, 0, 0);
  }

  // start next layer
  if(opt_count == local_nnz) {
    opt_count = 0;
    mw0_kernel();
    ARENA_spawn_task(KERNEL_LAYER1, ARENA_local_start, ARENA_local_end,
                     0, ARENA_local_rank, 0, 0);
  }

  return -1;//num_spawn;
}

int ARENA_kernel0_accum(int start, int end, int param) {
  ++opt_count;
//  cout<<"rank "<<ARENA_local_rank<<" ready to partial sum at local_start "<<start<<" aka global start "<<ARENA_local_start+start<<endl;
  for(int i=0; i<num_feature; ++i) {
//    cout<<"rank "<<ARENA_local_rank<<" buff_X["<<i<<"] "<<buff_X[i]<<" added into out0["<<start<<"]["<<i<<"]"<<endl; 
    out0[start][i] += buff_X[i];

//    cout<<"rank "<<ARENA_local_rank<<" recv_X["<<param<<"]["<<i<<"] "<<recv_X[param][i]<<endl; 
//    out0[start][i] += recv_X[param][i];
  }

  // start next layer
  if(opt_count == local_nnz) {
    opt_count = 0;
    mw0_kernel();
    ARENA_spawn_task(KERNEL_LAYER1, ARENA_local_start, ARENA_local_end,
                     0, ARENA_local_rank, 0, 0);
  }

  return -1;//num_spawn;
}

int cur_layer = 0;
int k_dim;
int ARENA_kernel1(int start, int end, int param) {
  cur_layer = 1;
  range = num_vertice/NODES;

  data_send_times[param] = 0;
  bool* sent_tag = new bool[NODES];
  for(int i=0; i<NODES; ++i) {
    sent_tag[i] = false;
  }
  // iterate across the value inside a specific row
  for(int i=local_ROW[param]; i<local_ROW[param+1]; ++i) {
    // if the index is inside my own data range, accumulate it locally
    if(local_COL[i] >= ARENA_local_start and
       local_COL[i] < ARENA_local_end) {
      for(int x=0; x<num_w0_out; ++x) {
//        cout<<"layer 1 rank "<<ARENA_local_rank<<" ready to self partial sum out1["<<param<<"]["<<x<<"] "<<out1[param][x]<<" add to out2["<<local_COL[i]-ARENA_local_start<<"]["<<x<<"] "<<out2[local_COL[i]-ARENA_local_start][x]<<endl;
        out2[local_COL[i]-ARENA_local_start][x] += out1[param][x];
      }
      // count the number of operation to guide the start of next layer
      ++opt_count;
    } else {
      // NOTE that ARENA does not allow send to one destination multiple times at one shot! Thus, we only send features to a destination once.
      int dest = local_COL[i]/range;
      if(!sent_tag[dest]) {
//        cout<<"layer 1 rank "<<ARENA_local_rank<<" spawn remote task at row "<<param<<" from "<<local_COL[i]<<" to "<<local_COL[i]+1<<" and send features"<<endl;
        sent_tag[dest] = true;
        ARENA_spawn_task(KERNEL_LAYER1_ACCUM, local_COL[i],
                         local_COL[i]+1, param+ARENA_local_start,
                         ARENA_local_rank, 0, num_w0_out);
        data_send_times[param] += 1;
      } else {
//        cout<<"layer 1 rank "<<ARENA_local_rank<<" spawn remote task at row "<<param<<" from "<<local_COL[i]<<" to "<<local_COL[i]+1<<" WITHOUT send features"<<endl;
        ARENA_spawn_task(KERNEL_LAYER1_ACCUM, local_COL[i],
                         local_COL[i]+1, param+ARENA_local_start,
                         ARENA_local_rank, 0, 0);
      }
      ARENA_remote_ask_start[local_COL[i]/range] = 0;
      ARENA_remote_ask_end[local_COL[i]/range] = num_w0_out;

    }
  }
  
  // iterating the rows by spawning new local tasks untill the boundary
  if(param < range-1) {
    ARENA_spawn_task(KERNEL_LAYER1, ARENA_local_start, ARENA_local_end,
                     param+1, ARENA_local_rank, 0, 0);
  }

  // start next layer
  if(opt_count == local_nnz) {
    opt_count = 0;
    mw1_kernel();
  }

  return -1;
}

int ARENA_kernel1_accum(int start, int end, int param) {
  ++opt_count;
//  cout<<"layer 1 rank "<<ARENA_local_rank<<" ready to partial sum at local_start "<<start<<" aka global start "<<ARENA_local_start+start<<endl;
  for(int i=0; i<num_w0_out; ++i) {
//    cout<<"layer 1 rank "<<ARENA_local_rank<<" buff_X["<<i<<"] "<<buff_X[i]<<" added into out0["<<start<<"]["<<i<<"]"<<endl; 
    out2[start][i] += buff_X[i];

//    cout<<"rank "<<ARENA_local_rank<<" recv_X["<<param<<"]["<<i<<"] "<<recv_X[param][i]<<endl; 
//    out0[start][i] += recv_X[param][i];
  }

  // start next layer
  if(opt_count == local_nnz) {
    opt_count = 0;
    mw1_kernel();
  }

  return -1;//num_spawn;
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
  ARENA_init_data_buff(num_feature, true);
  for(int x=0; x<NODES; ++x) {
    ARENA_remote_ask_buff[x] = new float[num_feature];
    ARENA_local_need_buff[x] = new float[num_feature];
  }
}

// ----------------------------------------------------------------------
// Main function. No need to change.
// ----------------------------------------------------------------------
int main(int argc, char *argv[]) {

  // Initialize global data start and end
  ARENA_init(argc, argv, NODES);

  ARENA_register(KERNEL_LAYER0, &ARENA_kernel0, true);
  ARENA_register(KERNEL_LAYER0_ACCUM, &ARENA_kernel0_accum, false);
  ARENA_register(KERNEL_LAYER1, &ARENA_kernel1, false);
  ARENA_register(KERNEL_LAYER1_ACCUM, &ARENA_kernel1_accum, false);

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
        cout<<"out0["<<i<<"]["<<j<<"]: "<<out0[i][j]<<"; ";
      }
      cout<<" ]"<<endl;
    }
    cout<<"[final] rank "<<ARENA_local_rank<<" out1: "<<endl;
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_w0_out-1; j<num_w0_out; ++j) {
        cout<<out1[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }

//    cout<<"[final] rank "<<ARENA_local_rank<<" trans_out1: "<<endl;
//    for(int i=num_w0_out-1; i<num_w0_out; ++i) {
//      cout<<"[ ";
//      for(int j=num_vertice/NODES-1; j<num_vertice/NODES; ++j) {
//        cout<<trans_out1[i][j]<<" ";
//      }
//      cout<<" ]"<<endl;
//    }

    cout<<"[final] rank "<<ARENA_local_rank<<" out2: "<<endl;
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_w0_out-1; j<num_w0_out; ++j) {
        cout<<out2[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }

    cout<<"[final] rank "<<ARENA_local_rank<<" out3: "<<endl;
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_w1_out-1; j<num_w1_out; ++j) {
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
int cur_row = 0;
int cur_send_times = 0;
bool flip = false;
void ARENA_load_data(int start, int end, float* buff) {
  float** temp_buff = local_X;
  if(cur_layer == 1) {
    temp_buff = out1;
  }
  if(cur_layer == 1 and !flip) {
    cur_row = 0;
    flip = true;
  }

  while(cur_send_times >= data_send_times[cur_row]) {
    cur_row += 1;
  }
  for(int i=start; i<end; ++i) {
    buff[i] = temp_buff[cur_row][i];
//    cout<<"buff["<<i<<"]: "<<buff[i]<<" ";
  }
//  cout<<endl;
  cur_send_times += 1;
  if(cur_send_times == data_send_times[cur_row]) {
    cur_send_times = 0;
    cur_row += 1;
  }
/*
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
*/
}

// ----------------------------------------------------------------------
// Receive data from remote nodes and store into local memory.
// TODO: user specified if necessary
// ----------------------------------------------------------------------
void ARENA_store_data(int start, int end, int source, float* buff) {
  int offset = source*num_vertice/NODES + data_recv_times[source];
  for(int i=start; i<end; ++i) {
    buff_X[i] = buff[i];
  }

//    // local_X[i][k_dim] = buff[i];
//    recv_X[offset][i] = buff[i];
//    cout<<"rank "<<ARENA_local_rank<<" received data "<<buff[i]<<" and store into recv_X["<<offset<<"]["<<i<<"]"<<endl;
//    // trans_X[k_dim][i] = buff[i];
//  }
//  data_recv_times[source] += 1;
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

  cout<<"[init recv_X] rank "<<ARENA_local_rank<<" : "<<endl;
  for(int i=0; i<num_vertice; ++i) {
    cout<<"[ ";
    for(int j=0; j<num_feature; ++j) {
      cout<<recv_X[i][j]<<" ";
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
