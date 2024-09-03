${\textsf{\color{magenta}This repo will be also available at }}$ https://github.com/ROCm/MAD ${\textsf{\color{magenta} soon}}$ 

# LLM inference performance validation with vLLM on the AMD Instinct MI300X accelerator

## Overview 🎉
--------

vLLM is a toolkit and library for large language model (LLM) inference and serving. It
deploys the PagedAttention algorithm, which reduces memory consumption
and increases throughput by leveraging dynamic key and value allocation
in GPU memory. vLLM also incorporates many recent LLM acceleration and
quantization algorithms. In addition, AMD implements high-performance custom
kernels and modules in vLLM to enhance performance further.

This Docker image packages vLLM with PyTorch for an AMD Instinct™ MI300X
accelerator. It includes:

-   ✅ ROCm™ 6.2
-   ✅ vLLM 0.4.3
-   ✅ PyTorch 2.4 
-   ✅ Tuning files (.csv format)

With this docker image, the users can quickly validate expected inference performance numbers on the MI 300 accelerator. 
We also provide tips and techniques so that users can get optimal performance with popular AI models.


## Reproducing benchmark results 🚀
-----------------------------

Use the following instructions to reproduce the benchmark results on an
MI300X accelerator with a prebuilt vLLM Docker image.

### NUMA balancing setting

To optimize performance, disable automatic NUMA balancing. Otherwise, the GPU
might hang until the periodic balancing is finalized. For further
details, refer to the [AMD Instinct MI300X system optimization](https://rocmdocs.amd.com/en/latest/how-to/system-optimization/mi300x.html#disable-numa-auto-balancing) guide.

```sh
# disable automatic NUMA balancing
sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
# check if NUMA balancing is disabled (returns 0 if disabled)
cat /proc/sys/kernel/numa_balancing
0
```

### Download the Docker image 🐳

The following command pulls the Docker image from Docker Hub and
launches a new Docker instance (*vllm\_mi300x*).

```sh
docker pull rocm/pytorch-private:20240828_exec_dashboard_unified_vllm_v7 # TODO: update to the final public image
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video -p 8080:8080 --shm-size 16G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --cap-add=SYS_PTRACE -v $(pwd):/workspace --env HUGGINGFACE_HUB_CACHE=/workspace --name unified_docker_vllm rocm/pytorch-private:20240828_exec_dashboard_unified_vllm_v7
```

### LLM performance settings

Some environment variables enhance the performance of the vLLM kernels
and PyTorch's tunableOp on the MI300X accelerator. 
The docker image is already preconfigured to include the performance settings. 
See the [AMD Instinct MI300X workload optimization](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html) guide for more information.

### Multiprocessing distributed executor

To optimize vLLM performance, We recommend using multiprocessing API server argument. Adding the `--distributed-executor-backend mp` does so.

### Copy the repository from GitHub

Copy the performance benchmarking scripts from GitHub to a local directory.

```sh
git clone https://github.com/seungrokj/unified_docker_benchmark_public # TODO: this repo will be also available at https://github.com/ROCm/MAD soon
cd unified_docker_benchmark_public
```

### Methodology

Use the following command and variables to run the benchmark tests.

#### Command

```sh
./vllm_benchmark_report.sh -s $test_option -m $model_repo -g $num_gpu -d $datatype
```

-   Note: The input sequence length, output sequence length, and tensor parallel (TP) are already configured. You don't need to specify them with this script.

-   Note: If you encounter this error, you need to pass your access-authorized huggingface token to the gated models.
```sh
OSError: You are trying to access a gated repo.

# pass your HF_TOKEN
export HF_TOKEN=$your_personal_hf_token
```

#### Variables

| Name         | Options                                 | Description                                      |
| ------------ | --------------------------------------- | ------------------------------------------------ |
| $test_option | latency                                 | Measure decoding token latency                   |
|              | throughput                              | Measure token generation throughput              |
|              | all                                     | Measure both throughput and latency              |
| $model_repo  | meta-llama/Meta-Llama-3.1-8B-Instruct   | Llama 3.1 8B                                     |
|              | meta-llama/Meta-Llama-3.1-70B-Instruct  | Llama 3.1 70B                                    |
|              | meta-llama/Meta-Llama-3.1-405B-Instruct | Llama 3.1 405B                                   |
|              | meta-llama/Llama-2-7b-chat-hf           | Llama 2 7B                                       |
|              | mistralai/Mistral-7B-Instruct-v0.3      | Mistral 7B                                       |
|              | Qwen/Qwen2-7B-Instruct                  | Qwen2 7B                                         |
|              | core42/jais-13b-chat                    | JAIS 13B                                         |
|              | core42/jais-30b-chat-v3                 | JAIS 30B                                         |
| $num_gpu     | 1 or 8                                  | Number of GPUs.                                  |
| $datatype    | float16                                 |                                                  |
                                                              

#### Run the benchmark tests on the MI300X accelerator 🏃

Here are some examples and the test results:

-   Benchmark example - latency
 
Use this command to benchmark the latency of the Llama 3.1 8B model on one GPU with the float16 data type.

```sh
./vllm_benchmark_report.sh -s latency -m meta-llama/Meta-Llama-3.1-8B-Instruct -g 1 -d float16
```

You can find the latency report at *./reports_float16/ Meta-Llama-3.1-8B-Instruct_latency_report.csv*.

-   Benchmark example - throughput

Use this command to benchmark the throughput of the Llama 3.1 8B model on one GPU with the fp16 data type.

```sh
./vllm_benchmark_report.sh -s throughput -m meta-llama/Meta-Llama-3.1-8B-Instruct -g 1 -d float16
```

You can find the throughput report at *./reports_float16/ Meta-Llama-3.1-8B-Instruct_throughput_report.csv*.

-   throughput\_tot = requests \* (**input lengths + output lengths**) / elapsed\_time

-   throughput\_gen = requests \* **output lengths** / elapsed\_time

## References 🔎
----------

For an overview of the optional performance features of vLLM with
ROCm software, see
<https://github.com/ROCm/vllm/blob/main/ROCm_performance.md>.

To learn more about the options for latency and throughput
benchmark scripts, see
<https://github.com/ROCm/vllm/tree/main/benchmarks>.

To learn how to run LLM models from Hugging Face or your own model, see the
[Using ROCm for AI](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/index.html) section of the ROCm documentation.

To learn how to optimize inference on LLMs, see the
[Fine-tuning LLMs and inference optimization](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/index.html) section of the ROCm documentation.

For a list of other ready-made Docker images for ROCm, see the 
[ROCm Docker image support matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/docker-image-support-matrix.html).

## Licensing Information ⚠️
---------------------

Your use of this application is subject to the terms of the applicable
component-level license identified below. To the extent any subcomponent
in this container requires an offer for corresponding source code, AMD
hereby makes such an offer for corresponding source code form, which
will be made available upon request. By accessing and using this
application, you are agreeing to fully comply with the terms of this
license. If you do not agree to the terms of this license, do not access
or use this application.

The application is provided in a container image format that includes
the following separate and independent components:

| Package | License                                          | URL                  |
| ------- | ------------------------------------------------ | -------------------- |
| Ubuntu  | Creative Commons CC-BY-SA Version 3.0 UK License | [Ubuntu Legal](https://ubuntu.com/legal) |
| ROCm    | Custom/MIT/Apache V2.0/UIUC OSL                  | [ROCm Licensing Terms](https://rocm.docs.amd.com/en/latest/about/license.html) |
| PyTorch | Modified BSD                                     | [PyTorch License](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| vLLM    | Apache License 2.0                               | [vLLM License](https://github.com/vllm-project/vllm/blob/main/LICENSE)  |

### Disclaimer

The information contained herein is for informational purposes only and
is subject to change without notice. In addition, any stated support is
planned and is also subject to change. While every precaution has been
taken in the preparation of this document, it may contain technical
inaccuracies, omissions and typographical errors, and AMD is under no
obligation to update or otherwise correct this information. Advanced
Micro Devices, Inc. makes no representations or warranties with respect
to the accuracy or completeness of the contents of this document, and
assumes no liability of any kind, including the implied warranties of
noninfringement, merchantability or fitness for purposes, with respect
to the operation or use of AMD hardware, software or other products
described herein. No license, including implied or arising by estoppel,
to any intellectual property rights is granted by this document. Terms
and limitations applicable to the purchase or use of AMD's products are
as set forth in a signed agreement between the parties or in AMD\'s
Standard Terms and Conditions of Sale.

### Notices and attribution

© 2024 Advanced Micro Devices, Inc. All rights reserved. AMD, the AMD
Arrow logo, Instinct, Radeon Instinct, ROCm, and combinations thereof
are trademarks of Advanced Micro Devices, Inc.

Docker and the Docker logo are trademarks or registered trademarks of
Docker, Inc. in the United States and/or other countries. Docker, Inc.
and other parties may also have trademark rights in other terms used
herein. Linux® is the registered trademark of Linus Torvalds in the U.S.
and other countries.    

All other trademarks and copyrights are property of their respective
owners and are only mentioned for informative purposes.   
