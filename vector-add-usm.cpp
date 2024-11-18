#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
#include <fstream>
#include <omp.h>
#include <thread>



using namespace sycl;


typedef enum {gpu, cpu, multithread} hardware;

//CHANGE parameter here
struct config
{
 size_t vector_size =1024*256; //define size as number of elements (4 byte int)
 int omp_threads =8;
 size_t kib=0;
 size_t mib=1024;
 bool usm=true;
 bool do_validation=true;
 hardware hw=gpu;  //cpu alternative
 std::string device_str = "cpu";
 std::string filename = "add_gpu.csv";
 float share_cpu =0.5f;
 size_t start_index=0;
 std::string processing_mode ="";
};

struct times
{
 double runtime_chrono_ms=0.f;
 double runtime_event_ms=0.f;
 double runtime_omp=0.f;
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
 * -s share cpu factor 0..1
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



double VectorAdd(queue &q, const int *a, const int *b, int *sum, size_t size) {

  range<1> num_items{size};
  auto e = q.parallel_for(num_items, [=](auto i) { sum[i] = a[i] + b[i]; });

 
  e.wait();
  return(e.template get_profiling_info<info::event_profiling::command_end>() -
       e.template get_profiling_info<info::event_profiling::command_start>());
}


void InitializeArray(int *a, size_t size, bool usm) {
  for (size_t i = 0; i < size; i++) a[i] = i;
}

 void print_to_file (config conf, times timer  )
  {

 std::fstream myfile(conf.filename,std::ios_base::app | std::ios_base::trunc);
    myfile.open(conf.filename);

    
    
     if(myfile.peek() == std::ifstream::traits_type::eof())
    {

        std::ofstream myfile_out(conf.filename);

        myfile_out << "benchmark;datasize;device;time_ms_event;time_ms_chrono;omp_threads;cpu_share;mode" << std::endl;


        myfile_out.close();

    }
     myfile.close();
    myfile.open(conf.filename);
    myfile.seekg (0, std::ios::end);

    myfile <<"usm_add"<<";"<< conf.vector_size
    <<";"<<conf.device_str
    <<";"<<timer.runtime_event_ms /1000000<<";"<<timer.runtime_chrono_ms/1000 
    <<";"<< conf.omp_threads
    <<";"<< conf.share_cpu
    <<";"<< conf.processing_mode
    <<std::endl;


  }

/**
 * print config parameter for benchmark to console
 */
  void printcfg (config conf)
  {
    std::cout<<"PRINT CONFIG"<<std::endl;
    std::cout<<"mode: "<<conf.hw<<std::endl;
    std::cout<<"kib: " <<conf.kib<<std::endl;
    std::cout<<"mib: " <<conf.mib<<std::endl;
    std::cout<<"vector_size: " <<conf.vector_size<<std::endl;
    std::cout<<"output filename: " <<conf.filename<<std::endl;
    std::cout<<"CPU Share: " <<conf.share_cpu<<std::endl;
    std::cout<<"GPU Start index: " <<conf.start_index<<std::endl;
    std::cout<<"omp threads: " <<conf.omp_threads<<std::endl;
    std::cout<<"Processing mode: " <<conf.processing_mode<<std::endl;
    

  }

/**
 * Validate benchmark by performing same operation on cpu
 * return false if data is different from benchmark
 * return true if no errors found
 */
  bool validate (int * a, int * b, int * sum_sequential, int * data_device, size_t size)
  {
    //perform operation on host
    for (size_t i = 0; i < size; i++) sum_sequential[i] = a[i] + b[i];

    // Verify that the two arrays are equal.
    for (size_t i = 0; i < size; i++) {
      if (data_device[i] != sum_sequential[i]) {
        std::cout << "Vector add failed on device. at index "<<i<<"\n";
        std::cout <<" device |  host "<< data_device[i]<<" "<< sum_sequential[i]<<std::endl;
        return false;
      }
    }
    return true;

  }

  //run openmp add on cpu
  void omp_add (int * a, int * b, int * sum_parallel, config conf)
  {
    int n_per_thread = conf.vector_size / conf.omp_threads;
    int i;
     #pragma omp parallel num_threads(conf.omp_threads)
  {
    #pragma omp parallel for shared(a, b, sum_parallel) private(i) schedule(dynamic, n_per_thread), 
        for( i=0; i<conf.start_index; i++) {
		sum_parallel[i] = a[i]+b[i];
        }
  }
  }

void benchmark(config conf, int * a_in, int * b_in, int * c_in, size_t vectorsize)
{
  conf.vector_size=vectorsize;
// split input data for gpu and cpu. start index is first gpu value
  //cpu calculates from 0 to start_index -1. gpu start-index to vector size-1
  conf.start_index = (conf.vector_size-1) * conf.share_cpu ; 
  if (conf.share_cpu >= 0.99f)
  {
    conf.start_index = conf.vector_size;
  }


  //printcfg(conf);
  
  //select CPU default, GPU else
  auto selector = sycl::cpu_selector_v;
  if(conf.hw == gpu)
   selector = sycl::gpu_selector_v;


   //allocate unified memory buffer
  try {
    queue q(selector,property::queue::enable_profiling{});

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
             << q.get_device().get_info<info::device::name>() << "\n";
  
    //allocate unified memory
    int *a = malloc_shared<int>(conf.vector_size, q);
    int *b = malloc_shared<int>(conf.vector_size, q);

    int *sum_sequential = malloc_shared<int>(conf.vector_size, q);
    int *sum_parallel = malloc_shared<int>(conf.vector_size, q);

  //exit if allocation failed
    if ((a == nullptr) || (b == nullptr) || (sum_sequential == nullptr) ||
        (sum_parallel == nullptr)) {
      if (a != nullptr) free(a, q);
      if (b != nullptr) free(b, q);
      if (sum_sequential != nullptr) free(sum_sequential, q);
      if (sum_parallel != nullptr) free(sum_parallel, q);

      std::cout << "Shared memory allocation failure.\n";
      exit(-1);
    }

    // Initialize input arrays with values from 0 to array_size - 1
    InitializeArray(a, conf.vector_size, true);
    InitializeArray(b, conf.vector_size, true);
    int i;

    //Copy over input arrays to unified mem
    for(int i =0; i < conf.vector_size;i++)
    {
      a[i] = a_in[i];
       b[i] =b_in[i];
    }

    times timer;
//warmup RUN!
  timer.runtime_event_ms =VectorAdd(q, a, b, sum_parallel, conf.vector_size);
   int n_per_thread = conf.vector_size / conf.omp_threads;
  
  auto total1 = std::chrono::steady_clock::now();
  auto total2 = std::chrono::steady_clock::now();
     
      //Co Processing
      if(conf.start_index>= 1 && conf.start_index <  conf.vector_size-1)
      {
       
        conf.processing_mode = "coprocessing";
        total1 = std::chrono::steady_clock::now();
std::thread tt(omp_add, a,b,sum_parallel,conf);
 timer.runtime_event_ms =VectorAdd(q, a+conf.start_index, b+conf.start_index, sum_parallel+conf.start_index, conf.vector_size-conf.start_index);
   tt.join();     
   total2 = std::chrono::steady_clock::now();
 timer.runtime_chrono_ms =std::chrono::duration_cast<std::chrono::microseconds>(total2 - total1).count();

      }
      //only Sycl
      else if (conf.start_index == 0)
      {
        conf.processing_mode = "Sycl only";
        total1 = std::chrono::steady_clock::now();
         timer.runtime_event_ms =VectorAdd(q, a, b, sum_parallel, conf.vector_size);
         total2 = std::chrono::steady_clock::now();
         timer.runtime_chrono_ms =std::chrono::duration_cast<std::chrono::microseconds>(total2 - total1).count();

      }
      //only OMP
      else if (conf.start_index >= conf.vector_size-1)
      {
        conf.processing_mode = "OpenMP only";
          total1 = std::chrono::steady_clock::now();
        std::thread tt(omp_add, a,b,sum_parallel,conf);
        tt.join();     
   total2 = std::chrono::steady_clock::now();
 timer.runtime_chrono_ms =std::chrono::duration_cast<std::chrono::microseconds>(total2 - total1).count();
 }
      


   if(!validate(a,b,sum_sequential,sum_parallel, conf.vector_size))
   {
   // return -1 if validation failed
   exit(-1);
   }
 


  print_to_file(conf,timer);


    free(a, q);
    free(b, q);
    free(sum_sequential, q);
    free(sum_parallel, q);
  } catch (exception const &e) {
    std::cout << "An exception is caught while adding two vectors.\n";
    std::terminate();
  }
}
   
int main(int argc, char* argv[]) {

  config conf = ParseInputParams (argc, argv);

  
  //set params, generate random in main. throw in data with pointers
  //default behavior half gpu half cpu
  
  int * in_a;
  int * in_b;
  int * out_c;
  size_t vector_size =1024*256;

  benchmark(conf,in_a,  in_b,  out_c,vector_size);

 // std::cout << "Vector add successfully completed on device.\n";
  return 0;

 

}
