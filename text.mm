#include <iostream>
#include <vector>
#include <thread>
#include <string>
#include <chrono>
#include <random>
#include <limits>
#include <cstdlib>  // for rand(), srand()
#include <ctime>    // for time()
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

// C++ CPU 实现
template<typename T>
T find_max(const std::vector<T>& arr) {
    T max_val = std::numeric_limits<T>::lowest();
    for (const auto& val : arr) {
        if (val > max_val) max_val = val;
    }
    return max_val;
}

template<typename T>
void cpu_worker(int trials, size_t array_size, int num_threads) {
    std::vector<std::thread> threads;
    int jobs_per_thread = trials / num_threads;
    int leftover = trials % num_threads;

    auto worker = [array_size](int jobs, unsigned int seed) {
        std::mt19937 gen(seed);
        if constexpr (std::is_integral<T>::value) {
            // 使用整数类型的分布
            std::uniform_int_distribution<T> dist(-10000, 10000);
            for (int i = 0; i < jobs; ++i) {
                std::vector<T> arr(array_size);
                for (auto& val : arr) val = dist(gen);
                find_max(arr);
            }
        } else {
            // 使用浮动类型的分布
            std::uniform_real_distribution<T> dist(-10000.0, 10000.0);
            for (int i = 0; i < jobs; ++i) {
                std::vector<T> arr(array_size);
                for (auto& val : arr) val = dist(gen);
                find_max(arr);
            }
        }
    };

    for (int i = 0; i < num_threads; ++i) {
        int jobs = jobs_per_thread + (i < leftover ? 1 : 0);
        unsigned int seed = static_cast<unsigned int>(std::time(nullptr)) + i;
        threads.emplace_back(worker, jobs, seed);
    }

    for (auto& t : threads) t.join();
}

#include <iostream>
#include <vector>
#include <cstdlib>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

// 数据生成函数
template<typename T>
void generate_data(std::vector<T>& data, size_t array_size, bool is_int) {
    if (is_int) {
        // 生成整数数据
        for (auto& val : data) {
            val = static_cast<T>(rand() % 20001 - 10000); // 范围 [-10000, 10000]
        }
    } else {
        // 生成浮动数据
        for (auto& val : data) {
            val = static_cast<T>((static_cast<double>(rand()) / RAND_MAX) * 20000.0 - 10000.0);
        }
    }
}

// GPU 实现（调用 Metal）
void gpu_run(int trials, size_t array_size, bool is_int) {
    // 获取 Metal 设备和队列
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSError *error = nil;

    // 直接将 Metal shader 代码嵌入到 C++ 代码中
    NSString *shader_src =
    @"#include <metal_stdlib>\n"
    "#include <metal_atomic>\n"
    "using namespace metal;\n"
    "kernel void find_max_kernel(const device float* input [[ buffer(0) ]], "
    "                            device atomic_float* result [[ buffer(1) ]], "
    "                            uint id [[ thread_position_in_grid ]]) {"
    "    float value = input[id];"
    "    float current_max = atomic_load_explicit(result, memory_order_relaxed);"
    "    while (value > current_max) {"
    "        if (atomic_compare_exchange_weak_explicit(result, &current_max, value, memory_order_relaxed, memory_order_relaxed)) {"
    "            break;"
    "        }"
    "    }"
    "}";

    // 创建 Metal 库
    id<MTLLibrary> library = [device newLibraryWithSource:shader_src options:nil error:&error];
    if (error) {
        std::cerr << error.localizedDescription.UTF8String << std::endl;
        return;
    }

    // 获取 Metal 函数
    id<MTLFunction> func = [library newFunctionWithName:@"find_max_kernel"];
    if (!func) {
        std::cerr << "Failed to load Metal function: find_max_kernel" << std::endl;
        return;
    }

    // 创建计算管道状态
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&error];
    if (error) {
        std::cerr << "Failed to create compute pipeline: " << error.localizedDescription << std::endl;
        return;
    }

    // 循环进行 trials 次测试
    for (int t = 0; t < trials; ++t) {
        std::vector<float> data(array_size);
        generate_data(data, array_size, is_int);

        // 创建 Metal buffer 并上传数据
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:data.data()
                                                        length:sizeof(float) * array_size
                                                       options:MTLResourceStorageModeShared];

        // 创建输出结果 buffer
        float result = -1e10;
        id<MTLBuffer> outputBuffer = [device newBufferWithBytes:&result
                                                         length:sizeof(float)
                                                        options:MTLResourceStorageModeShared];

        // 创建命令缓冲区和编码器
        id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];

        // 设置线程组大小
        MTLSize gridSize = MTLSizeMake(array_size, 1, 1);
        NSUInteger threadGroupSize = pipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > array_size) threadGroupSize = array_size;
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        // 调度线程
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        // 提交命令并等待完成
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // 获取并输出结果
        float *out = (float *)outputBuffer.contents;
        //std::cout << "GPU max = " << *out << std::endl;
    }
}
// 主函数
int main() {
    size_t array_size;
    int trials, num_threads;
    std::string type_choice, mode_choice;

    // 输入选择数据类型、线程数和计算模式
    std::cout << "Enter array size, trials, type (i/d), mode (cpu/gpu): ";
    std::cin >> array_size >> trials >> type_choice >> mode_choice;

    bool is_int = (type_choice == "i");
    bool use_gpu = (mode_choice == "gpu");

    auto start = std::chrono::high_resolution_clock::now();

    if (use_gpu) {
        gpu_run(trials, array_size, is_int);
    } else {
        std::cout << "Enter number of threads: ";
        std::cin >> num_threads;
        if (is_int) {
            cpu_worker<int>(trials, array_size, num_threads);
        } else {
            cpu_worker<double>(trials, array_size, num_threads);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    return 0;
}
