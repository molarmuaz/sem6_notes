## Platform Model

The OpenCL platform model defines the fundamental structure of an OpenCL system:

### Components:

1. **Host**:
    - Central control unit (typically a CPU)
    - Manages execution environment and dispatches commands
2. **OpenCL Devices**:
    - Computational units where parallel processing occurs
    - Contains multiple Compute Units (CUs)
3. **Compute Units (CUs)**:
    - Subdivisions of OpenCL devices
    - Contains multiple Processing Elements (PEs)
4. **Processing Elements (PEs)**:
    - Smallest computational units
    - Execute individual work-items

### Memory Types in OpenCL

| Memory Type  | Visible To                                   | Stored On                          | Speed                      | Use Case                                 | Keyword / Notes           |
| ------------ | -------------------------------------------- | ---------------------------------- | ----------------------- | ---------------------------------------- | ------------------------- |
| **Private**  | A single **thread** (like a local variable)  | On the core (registers)            | F                          | Temporary variables, counters            | Implicit (normal vars)    |
| **Local**    | All **threads in the same block/task group** | Shared block memory                                             | Shared cache between cooperating threads | `__local`                 |
| **Global**   | **All threads across all tasks** on the GPU  | GPU global memory                                               | Input/output buffers, large data         | `__global`                |
| **Constant** | All threads (**read-only**)                  | GPU global memory (special region) | F                          | Shared constants like lookup tables      | `__constant`              |
| **Host**     | CPU only                                     | System RAM          Very slow (needs copy)  copy)  copy)  copy) | Where data lives before/after GPU use    | Used via buffer transfers |

Basically: We have a host and devices. The host sets up the functions, the devices, contexts and queues.

The basic 5 steps to making a host program
- **Define** the platform (devices, contexts and queues)
- **Create and Build** the program (dynamic library for kernels)
- **Setup** memory objects
- **Define** the kernel (attach arguments to kernel function)
- **Submit** commands (transfer memory objects and execute kernels)

### Define
```c
// Get platform and device
cl::Platform platform;
cl::Device device;  
cl::Context context;
cl::CommandQueue queue;
```

**Platform:** Available Platforms like NVIDIA, AMD, etc

**Device:** Available Devices from those platforms e.g NVIDIA GeForce RTX 3080 GPU

**Context:** An environment/sandbox for devices to interact, manage memory, loading and executing     kernel functions, etc

**Command Queue:** Queue of commands for a device, includes; kernel executions, memory transfers  and syncronizations.


#### What it looks like in code
```cpp
std::vector<cl::Platform> platforms;
cl::Platform::get(&platforms); //Get platforms list
cl::Platform platform = platforms[0]; //Pick the first platform

std::vector<cl::Device> devices; //Creat device list
platform.getDevices(CL_DEVICE_TYPE_GPU, &devices); // Fill with platform's devices
cl::Device device = devices[0]; //Pick the first device

cl::Context context(device); // setup environment
cl::CommandQueue queue(context, device); // queue to send commands to GPU
```

#### A quicker way to setup (less safe and we don't pick the device)
```cpp
cl::Context context(CL_DEVICE_TYPE_DEFAULT);
/*
Also valid:
CL_DEVICE_TYPE_CPU,
CL_DEVICE_TYPE_GPU,
CL_DEVICE_TYPE_ACCELERATOR
*/

//It'll pick the first platform returned from getPlatforms, and add all the //devices from that into the context based on what device type we choose.
```

#### Queue Types:
- Each command queue operates on one device within a context only.
- We can configure command queues in different ways to organize which command gets executed first:
	- In order queue: Completed in order they appear in host program
	- Out of order queue: Completed in an order of its own choice.
		- **Why out of order:** Out-of-order queues let the GPU **optimize execution** by reordering commands if it can do so **more efficiently**, as long as data dependencies are preserved
##### How to setup out of order queueing:
```cpp
cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;

// Create queue
cl::CommandQueue queue(context, device, props);
```

#### Building the kernel program
When defining the source code of the kernel program, we can either define it within the program with a string literal (if it is a small program/instruction), or create a separate`.cl` file to write the function and then call it in the main program as a string. 

We package this kernel as a `cl::Program` object. It gets compiled at runtime for the target device. This object is used to make the kernel (`cl::Kernel`), set arguments and then it gets enqueued on the command queue.
**Simply put:**
[kernel.cl] → [read as string] → [cl::Program] → [cl::Kernel] → [enqueue for execution]

#### In-code (Slides did not go in-depth, so no need to memorize)
```cpp
// Once .cl file function defined.
//Use fstream functions to read and convert to string
using namespace std;
using namespace cl;

//Read the file as a string
ifstream kernelFile("reflect.cl");
string srcCode((istreambuf_iterator<char>(kernelFile)), {});

Program::Sources sources = { { srcCode.c_str(), srcCode.size() } };

//Compile kernel at runtime
cl::Program program(context, sources);

//Represent callable GPU function
cl::Kernel kernel(program, "horizontal_reflect");

//Set input/output buffers or images
kernel.setArg(0, srcImage);
kernel.setArg(1, dstImage);

//Schedule kernel for execution on GPU
queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(width, height));
// Args: Kernel, Offset (0 in this case i.e. NullRange), Global work size

queue.finish(); //Blocks the CPU until queue is empty again.

```




