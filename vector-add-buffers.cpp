//==============================================================
// Vector Add is the equivalent of a Hello, World! sample for data parallel
// programs. Building and running the sample verifies that your development
// environment is setup correctly and demonstrates the use of the core features
// of SYCL. This sample runs on both CPU and GPU (or FPGA). When run, it
// computes on both the CPU and offload device, then compares results. If the
// code executes on both CPU and offload device, the device name and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding SYCL Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// SYCL material used in the code sample:
// •	A one dimensional array of data.
// •	A device queue, buffer, accessor, and kernel.
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;

// num_repetitions: How many times to repeat the kernel invocation
size_t num_repetitions = 1;
size_t mib = 1;
// size in mib for 32 bit elements
size_t vector_size = 262144  ; 

typedef std::vector<int> IntVector; 

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

double  CopyVec(queue &q, const IntVector &a_vector, 
               IntVector &sum_parallel) {

  range<1> num_items{a_vector.size()};

  buffer a_buf(a_vector);
  
  buffer sum_buf(sum_parallel.data(), num_items);

  

    event e = q.submit([&](handler &h) {
      // Create an accessor for each buffer with access permission: read, write or
      // read/write. The accessor is a mean to access the memory in the buffer.
      accessor a(a_buf, h, read_only);
      
  
      // The sum_accessor is used to store (with write permission) the sum data.
      accessor sum(sum_buf, h, write_only, no_init);
  
      // Use parallel_for to run vector addition in parallel on device. This
      // executes the kernel.
      //    1st parameter is the number of work items.
      //    2nd parameter is the kernel, a lambda that specifies what to do per
      //    work item. The parameter of the lambda is the work item id.
      // SYCL supports unnamed lambda kernel by default.
      h.parallel_for(num_items, [=](auto i) { sum[i] = a[i]; });
    });
  
  // Wait until compute tasks on GPU done
  //q.wait();
 // return(e.template get_profiling_info<info::event_profiling::command_end>() -
  //     e.template get_profiling_info<info::event_profiling::command_start>());
  return -1.;
}

double  VectorAdd(queue &q, const IntVector &a_vector, const IntVector &b_vector,
               IntVector &sum_parallel) {
 
  range<1> num_items{a_vector.size()};
  
  buffer a_buf(a_vector);
  buffer b_buf(b_vector);
  buffer sum_buf(sum_parallel.data(), num_items);

  

    event e = q.submit([&](handler &h) {
     
      //sycl::property_list propList{sycl::property::queue::enable_profiling()};
      
      // Create an accessor for each buffer with access permission: read, write or
      // read/write. The accessor is a mean to access the memory in the buffer.
      accessor a(a_buf, h, read_only);
      accessor b(b_buf, h, read_only);
  
      // The sum_accessor is used to store (with write permission) the sum data.
      accessor sum(sum_buf, h, write_only, no_init);
  
      // Use parallel_for to run vector addition in parallel on device. This
      // executes the kernel.
      //    1st parameter is the number of work items.
      //    2nd parameter is the kernel, a lambda that specifies what to do per
      //    work item. The parameter of the lambda is the work item id.
      // SYCL supports unnamed lambda kernel by default.
      h.parallel_for(num_items, [=](auto i) { sum[i] = a[i] + b[i]; });
    });
  
  // Wait until compute tasks on GPU done
//  q.wait();
  return(e.template get_profiling_info<info::event_profiling::command_end>() -
       e.template get_profiling_info<info::event_profiling::command_start>());
   
}

//************************************
// Initialize the vector from 0 to vector_size - 1
//************************************
void InitializeVector(IntVector &a) {
  for (size_t i = 0; i < a.size(); i++) a.at(i) = i;
}

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {
  // Change num_repetitions if it was passed as argument
  std::string kernel = "nosync_add";
  if (argc > 2) kernel = argv[2];

  if(kernel == "add")
  {
    std::cout<<"nosync_add"<<std::endl;
  }

  else if (kernel == "copy")
  {
  std::cout<<"copy"<<std::endl;
  }

  



  // Change vector_size if it was passed as argument
  if (argc > 1) mib = std::stoi(argv[1]);
  // Create device selector for the device of your interest.
//vector_size = 256  ; 
  vector_size *= mib ; 
  
  auto selector = sycl::gpu_selector_v; //select gpu


  // Create vector objects with "vector_size" to store the input and output data.
  IntVector a, b, sum_sequential, sum_parallel;
  a.resize(vector_size);
  b.resize(vector_size);
  sum_sequential.resize(vector_size);
  sum_parallel.resize(vector_size);

  // Initialize input vectors with values from 0 to vector_size - 1
  InitializeVector(a);
  InitializeVector(b);
  double runtime_chrono=-1.f;
   double runtime_event=-1.f;

  
    queue q(selector,property::queue::enable_profiling{});
   

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << a.size() << "\n";

 
        auto t1 = std::chrono::steady_clock::now();
      runtime_event=VectorAdd(q, a, b, sum_parallel);
        auto t2 = std::chrono::steady_clock::now();
    
  
    
    
//output
  //std::string filename = kernel+std::to_string(mib)+"gpu.csv";
   std::string filename = "add_gpu.csv";
    std::fstream myfile(filename,std::ios_base::app | std::ios_base::trunc);
    myfile.open(filename);

    
    runtime_chrono=std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
     if(myfile.peek() == std::ifstream::traits_type::eof())
    {

        std::ofstream myfile_out(filename);

        myfile_out << "benchmark;datasize;time_ms_event;time_ms_chrono;throughput" << std::endl;


        myfile_out.close();

    }
     myfile.close();
    myfile.open(filename);
    myfile.seekg (0, std::ios::end);

    myfile <<kernel<<";"<< vector_size<<";"<<runtime_event /1000000<<";"<<runtime_chrono/1000 <<";-"<<std::endl;

  int indices[]{0, 1, 2, (static_cast<int>(a.size()) - 1)};
  constexpr size_t indices_size = sizeof(indices) / sizeof(int);

 
  a.clear();
  b.clear();
  sum_sequential.clear();
  sum_parallel.clear();

  return 0;
}
