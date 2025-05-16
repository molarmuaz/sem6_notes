We have a host and devices. The host sets up the functions, the devices, contexts and queues.

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
```c
std::vector<cl::Platform> platforms;
cl::Platform::get(&platforms); //Get platforms list
cl::Platform platform = platforms[0]; //Pick the first platform

std::vector<cl::Device> devices; //Creat device list
platform.getDevices(CL_DEVICE_TYPE_GPU, &devices); // Fill with platform's devices
cl::Device device = devices[0]; //Pick the first device

cl::Context context(device); // setup environment
cl::CommandQueue queue(context, device); // queue to send commands to GPU
```

