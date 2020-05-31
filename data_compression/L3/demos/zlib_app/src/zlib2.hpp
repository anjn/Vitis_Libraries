#pragma once
#include <queue>
#include <vector>
#include <list>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "xcl2.hpp"

namespace zlib2 {

constexpr uint32_t num_engines_per_kernel = 8;
constexpr uint32_t block_size_in_kb = 1024;
constexpr uint32_t c_ltree_size = 1024;
constexpr uint32_t c_dtree_size = 64;
constexpr uint32_t block_size = block_size_in_kb * 1024;
constexpr uint32_t host_buffer_size = block_size * num_engines_per_kernel;

struct deflate_worker;

struct deflate_job
{
  uint32_t id;
  uint8_t* data;
  uint32_t size;
  std::shared_ptr<deflate_worker> worker;
};

// Compute Unit
struct deflate_cu
{
  int index { 0 };

  // Event objects for controlling the order of job executions
  cl::Event ev_h2d_0;
  cl::Event ev_h2d_1;
  cl::Event ev_lz77;
  cl::Event ev_huffman;
};

struct deflate_worker
{
  deflate_cu& cu;

  cl::CommandQueue q;

  // Kernels
  cl::Kernel lz77_kernel;
  cl::Kernel huffman_kernel;

  // Device buffers
  cl::Buffer device_input;
  cl::Buffer device_lz77_output;
  cl::Buffer device_compress_size;
  cl::Buffer device_inblk_size;
  cl::Buffer device_dyn_ltree_freq;
  cl::Buffer device_dyn_dtree_freq;
  cl::Buffer device_output;

  std::vector<cl::Event> ev_read_data;

  // Host buffers
  template<typename T>
  using host_buffer = std::vector<T, zlib_aligned_allocator<T>>;
  template<typename T>
  size_t bytes(const host_buffer<T>& b) { return sizeof(uint32_t) * b.size(); }

  host_buffer<uint32_t> host_compress_size;
  host_buffer<uint32_t> host_inblk_size;

  deflate_worker(
    const cl::Device& device,
    const cl::Context& context,
    const cl::Program& program,
    deflate_cu& cu_
  ):
    cu(cu_)
  {
    // Create in-order queue
    q = cl::CommandQueue(context, device);

    // Create kernels
    std::string kernel_name = "xilLz77Compress:{xilLz77Compress_"s + std::to_string(cu.index + 1) + "}";
    lz77_kernel = cl::Kernel(program, kernel_name.c_str());
    kernel_name = "xilHuffmanKernel:{xilHuffmanKernel_"s + std::to_string(cu.index + 1) + "}";
    huffman_kernel = cl::Kernel(program, kernel_name.c_str());

    // Allocate host buffers
    host_compress_size.resize(num_engines_per_kernel);
    host_inblk_size   .resize(num_engines_per_kernel);

    // Create device buffers
    device_input          = cl::Buffer(context, CL_MEM_READ_ONLY, host_buffer_size);
    device_lz77_output    = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, host_buffer_size * 4);
    device_compress_size  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, bytes(host_compress_size), host_compress_size.data());
    device_inblk_size     = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, bytes(host_inblk_size), host_inblk_size.data());
    device_dyn_ltree_freq = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * c_ltree_size * num_engines_per_kernel);
    device_dyn_dtree_freq = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * c_dtree_size * num_engines_per_kernel);
    device_output         = cl::Buffer(context, CL_MEM_WRITE_ONLY, host_buffer_size * 2);
  }

  void compress(std::shared_ptr<deflate_job>& job)
  {
    auto size = job->size;

    // Input block size for each engine
    {
      uint32_t tmp_size = size;
      for (uint32_t i=0; i<num_engines_per_kernel; i++) {
        host_inblk_size[i] = std::min(block_size, tmp_size);
        tmp_size -= host_inblk_size[i];
      }
    }

    // Set args
    int narg = 0;
    lz77_kernel.setArg(narg++, device_input);
    lz77_kernel.setArg(narg++, device_lz77_output);
    lz77_kernel.setArg(narg++, device_compress_size);
    lz77_kernel.setArg(narg++, device_inblk_size);
    lz77_kernel.setArg(narg++, device_dyn_ltree_freq);
    lz77_kernel.setArg(narg++, device_dyn_dtree_freq);
    lz77_kernel.setArg(narg++, block_size_in_kb);
    lz77_kernel.setArg(narg++, size);

    // Set args
    narg = 0;
    huffman_kernel.setArg(narg++, device_lz77_output);
    huffman_kernel.setArg(narg++, device_dyn_ltree_freq);
    huffman_kernel.setArg(narg++, device_dyn_dtree_freq);
    huffman_kernel.setArg(narg++, device_output);
    huffman_kernel.setArg(narg++, device_compress_size);
    huffman_kernel.setArg(narg++, device_inblk_size);
    huffman_kernel.setArg(narg++, block_size_in_kb);
    huffman_kernel.setArg(narg++, size);

    // Copy host to device
    std::vector<cl::Event> wait_h2d;
    q.enqueueWriteBuffer(device_input, CL_FALSE, 0, size, job->data, &wait_h2d, &cu.ev_h2d_0);
    q.enqueueWriteBuffer(device_inblk_size, CL_FALSE, 0, bytes(host_inblk_size), host_inblk_size.data(), &wait_h2d, &cu.ev_h2d_1);

    // Invoke LZ77 kernel
    std::vector<cl::Event> wait_lz77;
    wait_lz77.push_back(cu.ev_h2d_0);
    wait_lz77.push_back(cu.ev_h2d_1);
    if (cu.ev_lz77() != NULL) wait_lz77.push_back(cu.ev_lz77);
    q.enqueueTask(lz77_kernel, &wait_lz77, &cu.ev_lz77);

    // Invoke Huffman kernel
    std::vector<cl::Event> wait_huffman = ev_read_data;
    wait_huffman.push_back(cu.ev_lz77);
    if (cu.ev_huffman() != nullptr) wait_huffman.push_back(cu.ev_huffman);
    q.enqueueTask(huffman_kernel, &wait_huffman, &cu.ev_huffman);

    // Copy device to host
    std::vector<cl::Event> wait_d2h;
    wait_d2h.push_back(cu.ev_huffman);
    q.enqueueReadBuffer(device_compress_size, CL_FALSE, 0, bytes(host_compress_size), host_compress_size.data(), &wait_d2h);
  }

  auto read_data(cl::CommandQueue& read_q, uint8_t* out)
  {
    // Wait compress
    q.finish();

    // Device to host
    ev_read_data.clear();
    for (uint32_t i=0; i<num_engines_per_kernel; i++)
    {
      if (host_inblk_size[i] == 0) break;

      cl::Event ev;

      read_q.enqueueReadBuffer(
        device_output,
        CL_FALSE,              // non-blocking
        block_size * i,        // offset
        host_compress_size[i], // size
        out,                   // ptr
        nullptr,
        &ev
      );

      ev_read_data.push_back(ev);

      out += host_compress_size[i];
    }

    return out;
  }
};

template<typename T>
struct mt_queue
{
  mutable std::mutex mtx;
  std::condition_variable cv;
  std::queue<T> items;
  bool stop_ { false };

  void push(const T& t)
  {
    std::unique_lock<std::mutex> lk(mtx);
    items.push(t);
    cv.notify_all();
  }

  T pop()
  {
    std::unique_lock<std::mutex> lk(mtx);

    cv.wait(lk, [this] { return stop_ || !items.empty(); });

    T t;
    if (stop_) return t;

    t = items.front();
    items.pop();

    return t;
  }

  void stop()
  {
    std::unique_lock<std::mutex> lk(mtx);
    stop_ = true;
    cv.notify_all();
  }
};

struct deflate
{
  // OpenCL
  std::vector<cl::Device> devices;
  std::vector<cl::Context> contexts;
  std::vector<cl::Program> programs;

  // CUs, Workers
  std::vector<deflate_cu> cus;
  mt_queue<std::shared_ptr<deflate_worker>> workers;

  // Jobs
  std::atomic<uint32_t> job_id;
  mt_queue<std::shared_ptr<deflate_job>> jobs;

  std::list<uint32_t> enqueued_jobs;
  mutable std::mutex mtx;
  std::condition_variable cv_enqueued_jobs;

  // Thread
  std::thread enqueue_compress_thread;

  deflate(): job_id(0) {}

  ~deflate()
  {
    workers.stop();
    jobs.stop();

    if (enqueue_compress_thread.joinable())
      enqueue_compress_thread.join();
  }

  void init(const std::string& xclbin_file)
  {
    // Init OpenCL
    devices = xcl::get_xil_devices();
    devices.resize(1);

    auto xclbin = xcl::read_binary_file(xclbin_file);
    cl::Program::Binaries bins{{xclbin.data(), xclbin.size()}};

    for (auto& device: devices) {
      auto context = cl::Context(device);
      auto program = cl::Program(context, {device}, bins);
      contexts.push_back(context);
      programs.push_back(program);
    }

    // Init workers
    const int num_workers_per_cu = 6;
    const int num_cus_per_device = 1;
    const int num_workers_per_device = num_cus_per_device * num_workers_per_cu;
    const int num_cus = num_cus_per_device * devices.size();
    const int num_workers = num_workers_per_device * devices.size();

    cus.resize(num_cus);
    for (int i=0; i<num_cus; i++) cus[i].index = i % num_cus_per_device;

    for (int i=0; i<num_workers; i++) {
      int device_index = (i/num_cus_per_device)%devices.size();
      auto device = devices[device_index];
      auto context = contexts[device_index];
      auto program = programs[device_index];
      auto cu = cus[i%num_cus];
      workers.push(std::make_shared<deflate_worker>(device, context, program, cu));
    }

    // Start thread
    enqueue_compress_thread = std::thread(&deflate::enqueue_compress, this);
  }

  void push_enqueued(const std::shared_ptr<deflate_job>& job)
  {
    std::unique_lock<std::mutex> lk(mtx);
    enqueued_jobs.push_back(job->id);
    cv_enqueued_jobs.notify_all();
  }

  // Function for thread
  void enqueue_compress()
  {
    while (true)
    {
      auto job = jobs.pop();
      if (!job) break;

      auto worker = workers.pop();
      if (!worker) break;

      // Assign job to worker
      job->worker = worker;
      worker->compress(job);

      push_enqueued(job);
    }
  }

  auto enqueue_compress_block(uint8_t* in_begin, uint8_t* in_end)
  {
    // Calculate job size
    intptr_t size = in_end - in_begin;
    size = std::min(size, (intptr_t) host_buffer_size);

    // Create job
    auto job = std::make_shared<deflate_job>();
    job->id = job_id.load(); job_id++;
    job->data = in_begin;
    job->size = size;
    job->worker = nullptr;

    jobs.push(job);

    return job;
  }

  auto copy_compressed_data(
    cl::CommandQueue& q,
    const std::shared_ptr<deflate_job>& job,
    uint8_t* out
  ) {
    // Wait job enqueued
    {
      std::unique_lock<std::mutex> lk(mtx);
      cv_enqueued_jobs.wait(lk, [this, &job] {
        for (auto it = enqueued_jobs.begin(); it != enqueued_jobs.end(); ++it) {
          if (*it == job->id) {
            enqueued_jobs.erase(it);
            return true;
          }
        }
        return false;
      });
    }

    // Read data
    out = job->worker->read_data(q, out);
    workers.push(job->worker);

    return out;
  }

  uint32_t compress(uint8_t* in, uint8_t* out, uint64_t in_size)
  {
    std::vector<std::shared_ptr<deflate_job>> jobs;
    jobs.reserve((in_size + host_buffer_size - 1) / host_buffer_size);

    // Enqueue jobs
    const auto in_end = in + in_size;
    while (in != in_end) {
      auto job = enqueue_compress_block(in, in_end);
      in += job->size;
      jobs.push_back(job);
    }

    // Get results
    auto read_q = cl::CommandQueue(contexts[0], devices[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    const auto out_begin = out;
    for (auto& job: jobs) {
      out = copy_compressed_data(read_q, job, out);
    }

    // zlib special block based on Z_SYNC_FLUSH
    *(out++) = 0x01;
    *(out++) = 0x00;
    *(out++) = 0x00;
    *(out++) = 0xff;
    *(out++) = 0xff;

    // Wait all read tasks done
    read_q.finish();

    return out - out_begin;
  }
};

}
