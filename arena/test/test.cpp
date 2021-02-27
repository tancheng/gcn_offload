#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
 
/**
 * @brief Illustrate how to get data from a target window.
 * @details This application consists of two MPI processes. MPI process 1
 * exposes a window containing an integer. MPI process 0 gets the value in it.
 * After the MPI_Get is issued, synchronisation takes place via MPI_Win_fence
 * and the MPI process 1 prints the value in its window.
 **/
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
 
    // Check that only 2 MPI processes are spawn
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if(comm_size != 4)
    {
        printf("This application is meant to be run with 4 MPI processes, not %d.\n", comm_size);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
 
    // Get my rank
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
 
    // Create the window
    int data_size = 100;
    int* window_buffer = new int[data_size];

    MPI_Win window;
    MPI_Win_create(window_buffer, data_size*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
    MPI_Win_fence(0, window);
 
    int* value_fetched = new int[data_size];
    for(int i=0; i<data_size; ++i) {
      value_fetched[i] = -1;
    }
    if(my_rank != 0) {
      // Fetch the value from the MPI process 1 window
      MPI_Get(value_fetched+my_rank*25, 25, MPI_INT, 0, my_rank*25, 25, MPI_INT, window);
    } else if(my_rank == 0) {
       for(int i=0; i<data_size; ++i) {
         window_buffer[i] = i;
       }
    }

    // Wait for the MPI_Get issued to complete before going any further
    MPI_Win_fence(0, window);
 
    if(my_rank != 0)
    {
      for(int i=my_rank*25; i<my_rank*25+25; ++i) {
        printf("[MPI process %d] %dth Value fetched from MPI process 0 window: %d.\n", my_rank, i, value_fetched[i]);
      }
    }
 
    // Destroy the window
    MPI_Win_free(&window);
 
    MPI_Finalize();
 
    return EXIT_SUCCESS;
}
