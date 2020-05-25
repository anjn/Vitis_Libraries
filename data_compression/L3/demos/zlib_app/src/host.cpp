/*
 * (c) Copyright 2019 Xilinx, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#include "zlib.hpp"
#include <fstream>
#include <vector>
#include <memory>
#include "cmdlineparser.h"

using namespace xf::compression;

const uint32_t num_engines_per_kernel = 8;
const uint32_t block_size_in_kb = 1024;
const uint32_t c_ltree_size = 1024;
const uint32_t c_dtree_size = 64;
const uint32_t block_size = block_size_in_kb * 1024;
const uint32_t host_buffer_size = block_size * num_engines_per_kernel;

struct compress_context
{
  cl::Context context;
  cl::CommandQueue q;

  // Kernels
  cl::Kernel lz77_kernel;
  cl::Kernel huffman_kernel;

  cl::Event ev_h2d_0;
  cl::Event ev_h2d_1;
  cl::Event ev_lz77;
  cl::Event ev_huffman;

  compress_context(
    const cl::Program& program,
    const cl::Context& context_,
    const cl::Device& device,
    int cu_index
  ) {
    context = context_;
    q = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    std::string kernel_name = "xilLz77Compress:{xilLz77Compress_"s + std::to_string(cu_index + 1) + "}";
    lz77_kernel = cl::Kernel(program, kernel_name.c_str());
    kernel_name = "xilHuffmanKernel:{xilHuffmanKernel_"s + std::to_string(cu_index + 1) + "}";
    huffman_kernel = cl::Kernel(program, kernel_name.c_str());
  }

  void finish() {
    q.finish();
  }
};

cl::Event ev_tmp0;
cl::Event ev_tmp1;

struct compress_worker
{
  std::shared_ptr<compress_context> c;
  //cl::CommandQueue q;

  // Kernels
  //cl::Kernel lz77_kernel;
  //cl::Kernel huffman_kernel;

  // Device buffers
  cl::Buffer device_input;
  cl::Buffer device_lz77_output;
  cl::Buffer device_compress_size;
  cl::Buffer device_inblk_size;
  cl::Buffer device_dyn_ltree_freq;
  cl::Buffer device_dyn_dtree_freq;
  cl::Buffer device_output;

  // Hos buffers
  template<typename T>
  using host_buffer = std::vector<T, zlib_aligned_allocator<T>>;
  template<typename T>
  size_t bytes(const host_buffer<T>& b) { return sizeof(uint32_t) * b.size(); }

  host_buffer<uint8_t>  host_input;
  host_buffer<uint8_t>  host_lz77_output;
  host_buffer<uint32_t> host_compress_size;
  host_buffer<uint32_t> host_inblk_size;
  host_buffer<uint8_t>  host_output;

  uint8_t* host_input_ptr { nullptr };

  cl::Event ev_read_size;
  cl::Event ev_read_data;

  // Status
  bool executing { false };

  compress_worker() {}

  void init(std::shared_ptr<compress_context>& c_)
  {
    c = c_;

    // Allocate host buffers
    host_input.        resize(host_buffer_size);
    host_compress_size.resize(num_engines_per_kernel);
    host_inblk_size.   resize(num_engines_per_kernel);
    host_output.       resize(host_buffer_size * 2);

    // Create device buffers
    device_input          = cl::Buffer(c->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, host_buffer_size, host_input.data());
    //device_input          = cl::Buffer(c->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, host_buffer_size);
    //device_input          = cl::Buffer(c->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, host_buffer_size, host_input.data());
    device_lz77_output    = cl::Buffer(c->context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, host_buffer_size * 4);
    device_compress_size  = cl::Buffer(c->context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(uint32_t) * num_engines_per_kernel, host_compress_size.data());
    device_inblk_size     = cl::Buffer(c->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(uint32_t) * num_engines_per_kernel, host_inblk_size.data());
    device_dyn_ltree_freq = cl::Buffer(c->context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * c_ltree_size * num_engines_per_kernel);
    device_dyn_dtree_freq = cl::Buffer(c->context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * c_dtree_size * num_engines_per_kernel);
    device_output         = cl::Buffer(c->context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, host_buffer_size * 2, host_output.data());
  }

  void finish() {
    if (host_input_ptr != nullptr) c->q.enqueueUnmapMemObject(device_input, host_input_ptr);
  }

  uint8_t* write_execute(uint8_t* in, uint8_t* in_end)
  {
    uint32_t size = in_end - in;
    if (size == 0) return in;
    if (size > host_buffer_size) size = host_buffer_size;

    // Input block size for each engine
    {
      uint32_t tmp_size = size;
      for (uint32_t i=0; i<num_engines_per_kernel; i++) {
        host_inblk_size[i] = std::min(block_size, tmp_size);
        tmp_size -= host_inblk_size[i];
      }
    }

    // Copy data to host buffer
    //std::memcpy(host_input.data(), in, size);
    
    //device_input          = cl::Buffer(c->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size, in);

    //
    int narg = 0;
    c->lz77_kernel.setArg(narg++, device_input);
    c->lz77_kernel.setArg(narg++, device_lz77_output);
    c->lz77_kernel.setArg(narg++, device_compress_size);
    c->lz77_kernel.setArg(narg++, device_inblk_size);
    c->lz77_kernel.setArg(narg++, device_dyn_ltree_freq);
    c->lz77_kernel.setArg(narg++, device_dyn_dtree_freq);
    c->lz77_kernel.setArg(narg++, block_size_in_kb);
    c->lz77_kernel.setArg(narg++, size);

    //
    narg = 0;
    c->huffman_kernel.setArg(narg++, device_lz77_output);
    c->huffman_kernel.setArg(narg++, device_dyn_ltree_freq);
    c->huffman_kernel.setArg(narg++, device_dyn_dtree_freq);
    c->huffman_kernel.setArg(narg++, device_output);
    c->huffman_kernel.setArg(narg++, device_compress_size);
    c->huffman_kernel.setArg(narg++, device_inblk_size);
    c->huffman_kernel.setArg(narg++, block_size_in_kb);
    c->huffman_kernel.setArg(narg++, size);

    //if (host_input_ptr == nullptr) host_input_ptr = (uint8_t*) c->q.enqueueMapBuffer(device_input, CL_TRUE, CL_MAP_WRITE, 0, host_buffer_size);
    //std::memcpy(host_input_ptr, in, size);

    // Host to device
    std::vector<cl::Event> wait_h2d;
    //if (c.ev_h2d_0() != NULL) wait_h2d.push_back(c.ev_h2d_0);
    //if (c.ev_h2d_1() != NULL) wait_h2d.push_back(c.ev_h2d_1);
    //c.q.enqueueMigrateMemObjects({device_input, device_inblk_size}, 0 /* 0 means from host*/, &wait_h2d, &c.ev_h2d);
    c->q.enqueueWriteBuffer(device_input, CL_FALSE, 0, size, in, &wait_h2d, &c->ev_h2d_0);
    c->q.enqueueWriteBuffer(device_inblk_size, CL_FALSE, 0, bytes(host_inblk_size), host_inblk_size.data(), &wait_h2d, &c->ev_h2d_1);

    //if (ev_tmp0() != NULL) wait_h2d.push_back(ev_tmp0);
    //if (ev_tmp1() != NULL) wait_h2d.push_back(ev_tmp1);
    //c->q.enqueueWriteBuffer(device_input, CL_FALSE, 0, size, in, &wait_h2d, &ev_tmp0);
    //c->q.enqueueWriteBuffer(device_inblk_size, CL_FALSE, 0, bytes(host_inblk_size), host_inblk_size.data(), &wait_h2d, &ev_tmp1);

    // Invoke LZ77 kernel
    std::vector<cl::Event> wait_lz77;
    wait_lz77.push_back(c->ev_h2d_0);
    wait_lz77.push_back(c->ev_h2d_1);
    //wait_lz77.push_back(ev_tmp0);
    //wait_lz77.push_back(ev_tmp1);
    if (c->ev_lz77() != NULL) wait_lz77.push_back(c->ev_lz77);
    c->q.enqueueTask(c->lz77_kernel, &wait_lz77, &c->ev_lz77);

    // Invoke Huffman kernel
    std::vector<cl::Event> wait_huffman;
    wait_huffman.push_back(c->ev_lz77);
    if (ev_read_data() != nullptr) wait_huffman.push_back(ev_read_data);
    if (c->ev_huffman() != nullptr) wait_huffman.push_back(c->ev_huffman);
    c->q.enqueueTask(c->huffman_kernel, &wait_huffman, &c->ev_huffman);

    // Device to host
    std::vector<cl::Event> wait_d2h;
    wait_d2h.push_back(c->ev_huffman);
    //c.q.enqueueMigrateMemObjects({device_compress_size}, CL_MIGRATE_MEM_OBJECT_HOST, &ev_huffman, &ev_read_size);
    c->q.enqueueReadBuffer(device_compress_size, CL_FALSE, 0, bytes(host_compress_size), host_compress_size.data(), &wait_d2h, &ev_read_size);

    executing = true;

    return in + size;
  }

  uint8_t* read(uint8_t* out)
  {
    if (!executing) return out;

    //q.finish();
    ev_read_size.wait();
    executing = false;

    // Device to host
    for (uint32_t i=0; i<num_engines_per_kernel; i++)
    {
      if (host_inblk_size[i] == 0) break;

      c->q.enqueueReadBuffer(
        device_output,
        CL_FALSE,              // non-blocking
        block_size * i,        // offset
        host_compress_size[i], // size
        out,                   // ptr
        nullptr,
        &ev_read_data
      );

      out += host_compress_size[i];
    }

    return out;
  }
};

const int num_contexts = 4;
const int num_workers = num_contexts * 8;

std::vector<std::shared_ptr<compress_context>> context;
std::vector<std::shared_ptr<compress_worker>> worker;

uint32_t compress2(xfZlib& zlib, uint8_t* in, uint8_t* out, uint32_t input_size)
{
  const auto out_begin = out;
  const auto in_end = in + input_size;

  //compress_context context(*zlib.m_context, zlib.m_device, *zlib.compress_kernel[0], *zlib.huffman_kernel[0]);
  //compress_context context[num_contexts];
  //for (int i=0; i<num_contexts; i++) {
  //  context[i].init(*zlib.m_program, *zlib.m_context, zlib.m_device, i);
  //}
  int worker_index = 0;
  auto curr_worker = [&]() -> compress_worker& { return *worker[worker_index]; };
  auto next_worker = [&](int i = 0) -> compress_worker& { return *worker[(worker_index + i) % num_workers]; };
  auto incr_worker = [&]() { worker_index = (worker_index + 1) % num_workers; };

  while (in != in_end) {
    for (int i=0; i<num_contexts; i++) {
      in  = next_worker(i).write_execute(in, in_end);
    }
    for (int i=0; i<num_contexts; i++) {
      out = next_worker(num_contexts).read(out);
      incr_worker();
    }
  }

  for (int i=0; i<num_workers; i++) {
    out = next_worker().read(out);
    next_worker().finish();
    incr_worker();
  }

  for (int i=0; i<num_contexts; i++) {
    context[i]->finish();
  }

  // zlib special block based on Z_SYNC_FLUSH
  *(out++) = 0x01;
  *(out++) = 0x00;
  *(out++) = 0x00;
  *(out++) = 0xff;
  *(out++) = 0xff;

  return out - out_begin;
}

void compress2_init(xfZlib& zlib)
{
  for (int i=0; i<num_contexts; i++) {
    context.push_back(std::make_shared<compress_context>(*zlib.m_program, *zlib.m_context, zlib.m_device, i));
  }
  for (int i=0; i<num_workers; i++) {
    worker.push_back(std::make_shared<compress_worker>());
    worker[i]->init(context[i%num_contexts]);
  }

  // warm-up
  //uint32_t input_size = 8*1024*1024 * num_contexts;
  //std::vector<uint8_t, zlib_aligned_allocator<uint8_t> > zlib_in(input_size);
  //std::vector<uint8_t, zlib_aligned_allocator<uint8_t> > zlib_out(input_size * 2);
  //compress2(zlib, zlib_in.data(), zlib_out.data(), input_size);
}

void compress2_free()
{
  worker.clear();
  context.clear();
}

void zlib_headers(std::string& inFile_name, std::ofstream& outFile, uint8_t* zip_out, uint32_t enbytes);

uint32_t compress_file2(xfZlib& zlib, std::string& inFile_name, std::string& outFile_name, uint64_t input_size) {
    std::ifstream inFile(inFile_name.c_str(), std::ifstream::binary);
    std::ofstream outFile(outFile_name.c_str(), std::ofstream::binary);

    if (!inFile) {
        std::cout << "Unable to open file";
        exit(1);
    }

    std::vector<uint8_t, zlib_aligned_allocator<uint8_t> > zlib_in(input_size);
    std::vector<uint8_t, zlib_aligned_allocator<uint8_t> > zlib_out(input_size * 2);

    inFile.read((char*)zlib_in.data(), input_size);

    compress2_init(zlib);
    // warm-up
    compress2(zlib, zlib_in.data(), zlib_out.data(), input_size);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    auto compress_API_start = std::chrono::high_resolution_clock::now();
    uint32_t enbytes = 0;

    // zlib Compress
    //enbytes = zlib.compress(zlib_in.data(), zlib_out.data(), input_size, HOST_BUFFER_SIZE);
    enbytes = compress2(zlib, zlib_in.data(), zlib_out.data(), input_size);

    auto compress_API_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::nano>(compress_API_end - compress_API_start);

    float throughput_in_mbps_1 = (float)input_size * 1000 / duration.count();
    std::cout << std::fixed << std::setprecision(3) << throughput_in_mbps_1;

    std::cout << std::flush;
    compress2_free();

    if (enbytes > 0) {
#ifdef GZIP_MODE
        // Pack gzip encoded stream .gz file
        gzip_headers(inFile_name, outFile, zlib_out.data(), enbytes);
#else
        // Pack zlib encoded stream .zlib file
        zlib_headers(inFile_name, outFile, zlib_out.data(), enbytes);
#endif
    }

    // Close file
    inFile.close();
    outFile.close();
    return enbytes;
}


void xil_validate(std::string& file_list, std::string& ext);

void xil_compress_decompress_list(std::string& file_list,
                                  std::string& ext1,
                                  std::string& ext2,
                                  int cu,
                                  std::string& single_bin,
                                  uint8_t max_cr,
                                  enum list_mode mode = COMP_DECOMP) {
    // Create xfZlib object
    xfZlib xlz(single_bin, max_cr, BOTH);

    if (mode != ONLY_DECOMPRESS) {
        std::cout << "--------------------------------------------------------------" << std::endl;
        std::cout << "                     Xilinx Zlib Compress                          " << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;

        std::cout << "\n";
        std::cout << "E2E(MBps)\tCR\t\tFile Size(MB)\t\tFile Name" << std::endl;
        std::cout << "\n";

        std::ifstream infilelist(file_list.c_str());
        std::string line;

        // Compress list of files
        // This loop does LZ4 compression on list
        // of files.
        while (std::getline(infilelist, line)) {
            std::ifstream inFile(line.c_str(), std::ifstream::binary);
            if (!inFile) {
                std::cout << "Unable to open file";
                exit(1);
            }

            uint64_t input_size = get_file_size(inFile);
            inFile.close();

            std::string compress_in = line;
            std::string compress_out = line;
            compress_out = compress_out + ext1;

            // Call Zlib compression
            //uint64_t enbytes = xlz.compress_file(compress_in, compress_out, input_size);
            uint64_t enbytes = compress_file2(xlz, compress_in, compress_out, input_size);

            std::cout << "\t\t" << (double)input_size / enbytes << "\t\t" << std::fixed << std::setprecision(3)
                      << (double)input_size / 1000000 << "\t\t\t" << compress_in << std::endl;
        }
    }

    // Decompress
    if (mode != ONLY_COMPRESS) {
        std::ifstream infilelist_dec(file_list.c_str());
        std::string line_dec;

        std::cout << "\n";
        std::cout << "--------------------------------------------------------------" << std::endl;
        std::cout << "                     Xilinx Zlib DeCompress                       " << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;
        std::cout << "\n";
        std::cout << "E2E(MBps)\tFile Size(MB)\t\tFile Name" << std::endl;
        std::cout << "\n";

        // Decompress list of files
        while (std::getline(infilelist_dec, line_dec)) {
            std::string file_line = line_dec;
            file_line = file_line + ext2;

            std::ifstream inFile_dec(file_line.c_str(), std::ifstream::binary);
            if (!inFile_dec) {
                std::cout << "Unable to open file";
                exit(1);
            }

            uint64_t input_size = get_file_size(inFile_dec);
            inFile_dec.close();

            std::string decompress_in = file_line;
            std::string decompress_out = file_line;
            decompress_out = decompress_out + ".orig";

            // Call Zlib decompression
            xlz.decompress_file(decompress_in, decompress_out, input_size, cu);

            std::cout << std::fixed << std::setprecision(3) << "\t\t" << (double)input_size / 1000000 << "\t\t"
                      << decompress_in << std::endl;
        } // While loop ends
    }
}

void xil_batch_verify(std::string& file_list, int cu, enum list_mode mode, std::string& single_bin, uint8_t max_cr) {
    std::string ext1;
    std::string ext2;

    // Xilinx ZLIB Compression
    ext1 = ".xe2xd.zlib";
    ext2 = ".xe2xd.zlib";

    xil_compress_decompress_list(file_list, ext1, ext2, cu, single_bin, max_cr, mode);

    // Validate
    std::cout << "\n";
    std::cout << "----------------------------------------------------------------------------------------"
              << std::endl;
    std::cout << "                       Validate: Xilinx Zlib Compress vs Xilinx Zlib Decompress           "
              << std::endl;
    std::cout << "----------------------------------------------------------------------------------------"
              << std::endl;
    std::string ext3 = ".xe2xd.zlib.orig";
    xil_validate(file_list, ext3);
}

void xil_decompress_top(std::string& decompress_mod, int cu, std::string& single_bin, uint8_t max_cr) {
    // Xilinx ZLIB object
    xfZlib xlz(single_bin, max_cr, DECOMP_ONLY);

    std::cout << std::fixed << std::setprecision(2) << "E2E(Mbps)\t\t:";

    std::ifstream inFile(decompress_mod.c_str(), std::ifstream::binary);
    if (!inFile) {
        std::cout << "Unable to open file";
        exit(1);
    }
    uint32_t input_size = get_file_size(inFile);

    std::string lz_decompress_in = decompress_mod;
    std::string lz_decompress_out = decompress_mod;
    lz_decompress_out = lz_decompress_out + ".raw";

    const char* sizes[] = {"B", "kB", "MB", "GB", "TB"};
    double len = input_size;
    int order = 0;
    while (len >= 1000) {
        order++;
        len = len / 1000;
    }

    // Call ZLIB compression
    // uint32_t enbytes =
    xlz.decompress_file(lz_decompress_in, lz_decompress_out, input_size, cu);
    std::cout << std::fixed << std::setprecision(3) << std::endl
              << "File Size(" << sizes[order] << ")\t\t:" << len << std::endl
              << "File Name\t\t:" << lz_decompress_in << std::endl;
}

void xil_compress_top(std::string& compress_mod, std::string& single_bin, uint8_t max_cr) {
    // Xilinx ZLIB object
    xfZlib xlz(single_bin, max_cr, COMP_ONLY);

    std::cout << std::fixed << std::setprecision(2) << "E2E(Mbps)\t\t:";

    std::ifstream inFile(compress_mod.c_str(), std::ifstream::binary);
    if (!inFile) {
        std::cout << "Unable to open file";
        exit(1);
    }
    uint32_t input_size = get_file_size(inFile);

    std::string lz_compress_in = compress_mod;
    std::string lz_compress_out = compress_mod;
    lz_compress_out = lz_compress_out + ".zlib";

    const char* sizes[] = {"B", "kB", "MB", "GB", "TB"};
    double len = input_size;
    int order = 0;
    while (len >= 1000) {
        order++;
        len = len / 1000;
    }

    // Call ZLIB compression
    //uint32_t enbytes = xlz.compress_file(lz_compress_in, lz_compress_out, input_size);
    uint64_t enbytes = compress_file2(xlz, lz_compress_in, lz_compress_out, input_size);

    std::cout.precision(3);
    std::cout << std::fixed << std::setprecision(2) << std::endl
              << "ZLIB_CR\t\t\t:" << (double)input_size / enbytes << std::endl
              << std::fixed << std::setprecision(3) << "File Size(" << sizes[order] << ")\t\t:" << len << std::endl
              << "File Name\t\t:" << lz_compress_in << std::endl;
    std::cout << "\n";
    std::cout << "Output Location: " << lz_compress_out.c_str() << std::endl;
}

void xil_validate(std::string& file_list, std::string& ext) {
    std::cout << "\n";
    std::cout << "Status\t\tFile Name" << std::endl;
    std::cout << "\n";

    std::ifstream infilelist_val(file_list.c_str());
    std::string line_val;

    while (std::getline(infilelist_val, line_val)) {
        std::string line_in = line_val;
        std::string line_out = line_in + ext;

        int ret = 0;
        // Validate input and output files
        ret = validate(line_in, line_out);
        if (ret == 0) {
            std::cout << (ret ? "FAILED\t" : "PASSED\t") << "\t" << line_in << std::endl;
        } else {
            std::cout << "Validation Failed" << line_out.c_str() << std::endl;
            exit(1);
        }
    }
}

void xilCompressDecompressTop(std::string& compress_decompress_mod, std::string& single_bin, uint8_t max_cr_val) {
    // Create xfZlib object
    xfZlib xlz(single_bin, max_cr_val, BOTH);

    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "                     Xilinx Zlib Compress                          " << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;

    std::cout << "\n";

    std::cout << std::fixed << std::setprecision(2) << "E2E(MBps)\t\t:";

    std::ifstream inFile(compress_decompress_mod.c_str(), std::ifstream::binary);
    if (!inFile) {
        std::cout << "Unable to open file";
        exit(1);
    }

    uint64_t input_size = get_file_size(inFile);
    inFile.close();

    std::string compress_in = compress_decompress_mod;
    std::string compress_out = compress_decompress_mod;
    compress_out = compress_out + ".zlib";

    const char* sizes[] = {"B", "kB", "MB", "GB", "TB"};
    double len = input_size;
    int order = 0;
    while (len >= 1000) {
        order++;
        len = len / 1000;
    }

    // Call Zlib compression
    //uint64_t enbytes = xlz.compress_file(compress_in, compress_out, input_size);
    uint64_t enbytes = compress_file2(xlz, compress_in, compress_out, input_size);
    std::cout << std::fixed << std::setprecision(2) << std::endl
              << "CR\t\t\t:" << (double)input_size / enbytes << std::endl
              << std::fixed << std::setprecision(3) << "File Size(" << sizes[order] << ")\t\t:" << len << std::endl
              << "File Name\t\t:" << compress_in << std::endl;

    // Decompress

    std::cout << "\n";
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "                     Xilinx Zlib DeCompress                       " << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "\n";

    std::cout << std::fixed << std::setprecision(2) << "E2E(MBps)\t\t:";

    // Decompress list of files

    std::string lz_decompress_in = compress_decompress_mod + ".zlib";
    std::string lz_decompress_out = compress_decompress_mod;
    lz_decompress_out = lz_decompress_in + ".orig";

    std::ifstream inFile_dec(lz_decompress_in.c_str(), std::ifstream::binary);
    if (!inFile_dec) {
        std::cout << "Unable to open file";
        exit(1);
    }

    input_size = get_file_size(inFile_dec);
    inFile_dec.close();

    // Call Zlib decompression
    xlz.decompress_file(lz_decompress_in, lz_decompress_out, input_size, 0);

    len = input_size;
    order = 0;
    while (len >= 1000) {
        order++;
        len = len / 1000;
    }

    std::cout << std::fixed << std::setprecision(2) << std::endl
              << std::fixed << std::setprecision(3) << "File Size(" << sizes[order] << ")\t\t:" << len << std::endl
              << "File Name\t\t:" << lz_decompress_in << std::endl;

    // Validate
    std::cout << "\n";

    std::string inputFile = compress_decompress_mod;
    std::string outputFile = compress_decompress_mod + ".zlib" + ".orig";
    int ret = validate(inputFile, outputFile);
    if (ret == 0) {
        std::cout << (ret ? "FAILED\t" : "PASSED\t") << "\t" << inputFile << std::endl;
    } else {
        std::cout << "Validation Failed" << outputFile.c_str() << std::endl;
    }
}
int main(int argc, char* argv[]) {
    int cu_run;
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--compress", "-c", "Compress", "");
    parser.addSwitch("--decompress", "-d", "DeCompress", "");
    parser.addSwitch("--single_xclbin", "-sx", "Single XCLBIN", "single");
    parser.addSwitch("--compress_decompress", "-v", "Compress Decompress", "");

    parser.addSwitch("--file_list", "-l", "List of Input Files", "");
    parser.addSwitch("--cu", "-k", "CU", "0");
    parser.addSwitch("--max_cr", "-mcr", "Maximum CR", "10");
    parser.parse(argc, argv);

    std::string compress_mod = parser.value("compress");
    std::string filelist = parser.value("file_list");
    std::string decompress_mod = parser.value("decompress");
    std::string single_bin = parser.value("single_xclbin");
    std::string compress_decompress_mod = parser.value("compress_decompress");
    std::string cu = parser.value("cu");
    std::string mcr = parser.value("max_cr");

    uint8_t max_cr_val = 0;
    if (!(mcr.empty())) {
        max_cr_val = atoi(mcr.c_str());
    } else {
        // Default block size
        max_cr_val = MAX_CR;
    }

    if (cu.empty()) {
        printf("please give -k option for cu\n");
        exit(0);
    } else {
        cu_run = atoi(cu.c_str());
    }

    if (!compress_decompress_mod.empty()) xilCompressDecompressTop(compress_decompress_mod, single_bin, max_cr_val);

    if (!filelist.empty()) {
        list_mode lMode;
        // "-l" - List of files
        if (!compress_mod.empty()) {
            lMode = ONLY_COMPRESS;
        } else if (!decompress_mod.empty()) {
            lMode = ONLY_DECOMPRESS;
        } else {
            lMode = COMP_DECOMP;
        }
        xil_batch_verify(filelist, cu_run, lMode, single_bin, max_cr_val);
    } else if (!compress_mod.empty()) {
        // "-c" - Compress Mode
        xil_compress_top(compress_mod, single_bin, max_cr_val);
    } else if (!decompress_mod.empty())
        // "-d" - DeCompress Mode
        xil_decompress_top(decompress_mod, cu_run, single_bin, max_cr_val);
}
