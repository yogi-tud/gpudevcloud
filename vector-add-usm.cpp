#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
#include <fstream>
#include <omp.h>



using namespace sycl;


typedef enum {gpu, cpu, multithread} hardware;


struct config
{
 size_t vector_size =256; //1kib for 4 byte data size
 int omp_threads =8;
 size_t kib=0;
 size_t mib=0;
 bool usm=true;
 bool do_validation=true;
 hardware hw=cpu;
 std::string device_str = "cpu";
 std::string filename = "add_gpu.csv";
 float share_cpu =100.f;
};

struct times
{
 double runtime_chrono_ms;
 double runtime_event_ms;
 double runtime_omp;
};

// Array size for this example.
size_t mib = 1;
size_t vector_size = 262144 ;

size_t  gpu_percent = 0;

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

/**
 * -k size in KiB
 * -m size in MiB
 * -d device sycl cpu or gpu
 * --nv no validation
 * -o output filename
 * -s share cpu [0 .. 100%]
 * -omp openmp threads int
 */
config ParseInputParams (int argc, char** argv)
{
  config conf;
 int w_argc = argc - 1; // remaining arg count
    while (w_argc > 0) {
        char* w_arg = argv[argc - (w_argc--)]; // working arg
        char* n_arg = (w_argc > 0) ? argv[argc - w_argc] : NULL; // next arg

        if (strcmp(w_arg, "--nv") == 0) {
            w_argc--;
            conf.do_validation = false;
        
        }

        else if (strcmp(w_arg, "-k") == 0) {
            w_argc--;
            size_t kib = atoi(n_arg);
            if (kib == 0) {
                kib = 1;
            }
            conf.kib = kib;
            conf.vector_size = conf.vector_size *conf.kib;
        }

        else if (strcmp(w_arg, "-m") == 0) {
            w_argc--;
            size_t mib = atoi(n_arg);
            if (mib == 0) {
                mib = 1;
            }
            conf.mib = mib;
            conf.vector_size = conf.vector_size *1024 * conf.mib;
        }
        else if (strcmp(w_arg, "-omp") == 0) {
            w_argc--;
            size_t omp = atoi(n_arg);
            if (omp == 0) {
                omp = 1;
            }
            conf.omp_threads = omp;
            
        }

         else if (strcmp(w_arg, "-d") == 0) {
            w_argc--;
            std::string mode_input = n_arg;
            
            if(mode_input.compare("cpu") ==0)
             {

            conf.hw = cpu;
            conf.device_str ="cpu";
            }

            if(mode_input.compare("gpu") ==0)
            {
            conf.hw = gpu;
            conf.device_str ="gpu";
            }
            

            
        }

        else if (strcmp(w_arg, "-o") == 0) {
            w_argc--;
            std::string ofile = n_arg;
            
            conf.filename = ofile;
        }

        
        else if (strcmp(w_arg, "-s") == 0) {
            w_argc--;
            float share_cpu  = atof(n_arg);
            
            
            conf.share_cpu = share_cpu;
        }

}

return conf;
}


//************************************
// Vector add in SYCL on device: returns sum in 4th parameter "sum".
//************************************
double VectorAdd(queue &q, const int *a, const int *b, int *sum, size_t size) {
  // Create the range object for the arrays.
  range<1> num_items{size};

  // Use parallel_for to run vector addition in parallel on device. This
  // executes the kernel.
  //    1st parameter is the number of work items.
  //    2nd parameter is the kernel, a lambda that specifies what to do per
  //    work item. the parameter of the lambda is the work item id.
  // SYCL supports unnamed lambda kernel by default.
  auto e = q.parallel_for(num_items, [=](auto i) { sum[i] = a[i] + b[i]; });

  // q.parallel_for() is an asynchronous call. SYCL runtime enqueues and runs
  // the kernel asynchronously. Wait for the asynchronous call to complete.
  e.wait();
  return(e.template get_profiling_info<info::event_profiling::command_end>() -
       e.template get_profiling_info<info::event_profiling::command_start>());
}


/**
 * init array
 * 
 */
void InitializeArray(int *a, size_t size, bool usm) {
  for (size_t i = 0; i < size; i++) a[i] = i;
}
 void print_to_file (  )
  {




  }

  void printcfg (config conf)
  {
    std::cout<<"PRINT CONFIG"<<std::endl;
    std::cout<<"mode: "<<conf.hw<<std::endl;
    std::cout<<"kib: " <<conf.kib<<std::endl;
    std::cout<<"mib: " <<conf.mib<<std::endl;
    std::cout<<"vector_size: " <<conf.vector_size<<std::endl;
    std::cout<<"output filename: " <<conf.filename<<std::endl;
    std::cout<<"CPU Share: " <<conf.share_cpu<<std::endl;
    std::cout<<"omp threads: " <<conf.omp_threads<<std::endl;
    

  }

  bool validate (int * a, int * b, int * sum_sequential, int * data_device, size_t size)
  {
    //perform operation on host
    for (size_t i = 0; i < size; i++) sum_sequential[i] = a[i] + b[i];

    // Verify that the two arrays are equal.
    for (size_t i = 0; i < size; i++) {
      if (data_device[i] != sum_sequential[i]) {
        std::cout << "Vector add failed on device. at index "<<i<<"\n";
        return false;
      }
    }
    return true;

  }

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {

  config conf = ParseInputParams (argc, argv);
  printcfg(conf);
  

 
  auto selector = sycl::cpu_selector_v;

  if(conf.hw == gpu)
   selector = sycl::gpu_selector_v;



  try {
    queue q(selector,property::queue::enable_profiling{});

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << conf.vector_size << "\n";

    // Create arrays with "array_size" to store input and output data. Allocate
    // unified shared memory so that both CPU and device can access them.
    int *a = malloc_shared<int>(conf.vector_size, q);
    int *b = malloc_shared<int>(conf.vector_size, q);
    int *sum_sequential = malloc_shared<int>(conf.vector_size, q);
    int *sum_parallel = malloc_shared<int>(conf.vector_size, q);

    if ((a == nullptr) || (b == nullptr) || (sum_sequential == nullptr) ||
        (sum_parallel == nullptr)) {
      if (a != nullptr) free(a, q);
      if (b != nullptr) free(b, q);
      if (sum_sequential != nullptr) free(sum_sequential, q);
      if (sum_parallel != nullptr) free(sum_parallel, q);

      std::cout << "Shared memory allocation failure.\n";
      return -1;
    }

    // Initialize input arrays with values from 0 to array_size - 1
    InitializeArray(a, conf.vector_size, true);
    InitializeArray(b, conf.vector_size, true);
    int i;
    double runtime_event=-1.f;
    double runtime_chrono=-1.f;
    double runtime_omp =-1.f;
    times timer;

   int n_per_thread = vector_size / conf.omp_threads;

      auto t1 = std::chrono::steady_clock::now();
       #pragma omp parallel num_threads(conf.omp_threads)
  {
    #pragma omp parallel for shared(a, b, sum_parallel) private(i) schedule(dynamic, n_per_thread), 
        for( i=0; i<conf.vector_size; i++) {
		sum_parallel[i] = a[i]+b[i];
        }
  }
  auto t2 = std::chrono::steady_clock::now();
  timer.runtime_omp =std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  

 t1 = std::chrono::steady_clock::now();
 timer.runtime_event_ms =VectorAdd(q, a, b, sum_parallel, conf.vector_size);
    t2 = std::chrono::steady_clock::now();
   timer.runtime_chrono_ms =std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    // Compute the sum of two arrays in sequential for validation.
    
 
    
   
  

    
    std::fstream myfile(conf.filename,std::ios_base::app | std::ios_base::trunc);
    myfile.open(conf.filename);

    
    runtime_chrono=std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
     if(myfile.peek() == std::ifstream::traits_type::eof())
    {

        std::ofstream myfile_out(conf.filename);

        myfile_out << "benchmark;datasize;device;time_ms_event;time_ms_chrono;time_ms_omp_chrono;omp_threads" << std::endl;


        myfile_out.close();

    }
     myfile.close();
    myfile.open(conf.filename);
    myfile.seekg (0, std::ios::end);

    myfile <<"usm_add"<<";"<< conf.vector_size
    <<";"<<conf.device_str
    <<";"<<timer.runtime_event_ms /1000000<<";"<<timer.runtime_chrono_ms/1000 
    <<";"<< timer.runtime_omp/1000 
    <<";"<< conf.omp_threads
    
    <<";"<<std::endl;



    free(a, q);
    free(b, q);
    free(sum_sequential, q);
    free(sum_parallel, q);
  } catch (exception const &e) {
    std::cout << "An exception is caught while adding two vectors.\n";
    std::terminate();
  }

  std::cout << "Vector add successfully completed on device.\n";
  return 0;

 

}
