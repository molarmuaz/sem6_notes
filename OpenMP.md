
# OpenMP
Framework that does most of the parallelization work we previously had to do in `pthreads` itself. We make API calls (#pragma...) and it does the work of creating and managing threads and mutexes. In threads we can just call the thread creation above the area of code we need it in (for e.g In a for loop), and it does the task of managing everything itself.

OpenMP is used in shared memory models i.e. all threads have access to the same shared memory.

#### Goals:
- **Standardization:** A standard model to be used seamlessly on different machines and architectures.
- **Thread-based:** No need to worry about low level threading code. It does that work itself
- **Cross-Platform:** Supports multiple operating systems and compilers.
- **User-directed:** The user explicitly defines what parts of code are to be parallelized.

#### OpenMP and MPI
OpenMP and MPI are used together as well in hybrid programming models like the bus interconnection model

**Bus Interconnect:** All processes have access to the same memory space, but when data is exchanged between distributed systems, a network or interconnect is used.

#### Syntax
We use directives and clauses to indicate OpenMP commands.

include the library `omp.h`.

OpenMP commands start with `#pragma omp <directive>`. The directive can be parallel, for, sections, etc.


This command acts as an entry point for parallel execution in OpenMP.

Commonly used clauses.

- `if (expression)`
	Only runs the region in parallel if expression is true, else sequentially.
	e.g `#pragma omp parallel if(x>3)`

- `num_threads(n)`
	defines how many threads to use
	e.g. `#pragma omp num_threads(4)`

- `private(list)`
	each threads gets its own copy of listed variables
	e.g.
	```c
	int i;
	#pragma omp parallel private(i)
	```

- `shared(list)`
	all threads share the variables (changing it in one threads it globally). Can cause data races if not synchronized. To overcome that we can use OpenMP's reduce
	```c
	int arr[10];
	#pragma omp parallel shared(arr)
	```


#### Setting thread count

Other than `#pragma omp parallel num_threads(n)`, we have two more commonly used methods of setting thread count:
- `omp_set_num_threads(n)` - we use this function in code. However if num_threads(n) is used in the directive. it will take precedence over this function.
- `OMP_NUM_THREADS` - we use at run-time to define to the compiler how many threads to use. num_threads(n) has precedence over this as well.

==*Note: Only if(exp) has precedence over num_threads(n) in that if exp is false, only 1 thread will be used.*==

Lastly, if we do not define the number of threads, there is a default thread count that the compiler will use which is usually the number of available CPU cores.

#### Some commonly used directives with their functionality
## Common OpenMP Directives

| Directive                   | Functionality                                                                                             |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| `#pragma omp parallel`      | Starts a parallel region; spawns threads to execute the code block                                        |
| `#pragma omp for`           | Distributes loop iterations among threads (must be inside a `parallel` block or combined with `parallel`) |
| `#pragma omp parallel for`  | Combines thread creation and loop parallelization in one directive                                        |
| `#pragma omp sections`      | Allows different threads to run different code blocks in parallel                                         |
| `#pragma omp section`       | Defines a single block inside `sections` to be run by one thread                                          |
| `#pragma omp single`        | Only one thread (not necessarily the master) executes the block                                           |
| `#pragma omp master`        | Only the master thread (thread 0) executes the block                                                      |
| `#pragma omp critical`      | Ensures only one thread executes a critical section at a time (avoids data races)                         |
| `#pragma omp atomic`        | Applies atomic operation for a single memory location update (more efficient than `critical`)             |
| `#pragma omp barrier`       | All threads wait here until each has reached this point                                                   |
| `#pragma omp task`          | Defines a task that can be run asynchronously by threads                                                  |
| `#pragma omp flush`         | Forces memory consistency across threads                                                                  |
| `#pragma omp taskwait`      | Waits for all child tasks to complete                                                                     |
| `#pragma omp threadprivate` | Makes a global variable private to each thread                                                            |

#### firstprivate and lastprivate
`#pragma omp firstprivate` - is used when we want the private variable of all the threads to have same initial value as the first thread.
`#pragma omp lastprivate` - is used when we want the global variable to have the same value as the last thread's value.
#### threadprivate vs private

the variable values stay in memory even if you end the parallel area in threadprivate and so if you call parallel area again those values will be there. This does not happen with private.


`omp_get_thread_num()` - gives the id of current thread.

#### Work-Sharing Constructs in OpenMP

Work-sharing constructs in OpenMP are used to **distribute work** (typically loops or sections of code) among threads in a parallel region. Instead of each thread performing the same work or the programmer manually splitting the work, OpenMP allows us to specify how work is divided among threads.

- #### **Thread Team and IDs**:
    
    - In OpenMP, all threads executing in parallel belong to a "team."
        
    - Each thread in the team is assigned a **unique ID**, with the **master thread** (the first thread) having ID 0.
        
    - You can use the function `omp_get_thread_num()` to get the ID of the current thread.
        
- #### **Can We Distribute Tasks?**
    
    - Yes! Work-sharing constructs are used to **distribute tasks** among the threads within the parallel region.
        
    - They **do not launch new threads**; they only divide the work among the threads in the team.

#### `#pragma omp sections` and `section`

###### `#pragma omp sections`:

Defines a **block** that contains **multiple independent tasks**.

###### `#pragma omp section`:

Each `section` inside `sections` is assigned to **one thread**.

Note: Threads and sections have 1:1 ratio. For every section one thread does the work i.e if there are more threads than sections, the extra threads will do nothing. And in the case of vice versa the threads will divide and work in a round robin fashion

#### Syntax:
```c
#pragma omp parallel
{
    #pragma omp sections
    {
        #pragma omp section
        {
            printf("Thread %d does Task A\n", omp_get_thread_num());
        }

        #pragma omp section
        {
            printf("Thread %d does Task B\n", omp_get_thread_num());
        }

        #pragma omp section
        {
            printf("Thread %d does Task C\n", omp_get_thread_num());
        }
    }
}
```

## Race Problem

We already know about the problems caused by accessing same shared data in certain cases. In pthreads we used to use mutexes for it, where we initiated the lock before playing around with shared data. It is even simpler in OpenMP. To enter a critical section we use `#pragma omp critical`. Furthermore, we can also use `#pragma omp master` to give the access only to the master thread.

We also use `#pragma omp barrier` to allow all threads to get up to speed just like we did in MPI Advance. 

Also similar to MPI Advance we use Reduction for standard tasks such as summation, AND/OR, max or min, etc.

Syntax: `#pragma omp parallel reduce(max:winner)`. Using reduction automatically handles all the inner intricacies itself.


#### How reduction can help:

**Situation**: want to sum all the numbers in an array.
```c
int sum = 0;
#pragma omp parallel for
for(int i = 0; i < n; i++) {
    sum += arr[i]; // 🔥 Race condition here!
}
```

This could very easily turn into a race problem because of a shared variable used across threads.

**Solution:**
```c
int sum = 0;

#pragma omp parallel for reduction(+:sum)
for(int i = 0; i < n; i++) {
    sum += arr[i];  // ✅ Thread-safe accumulation
}
```

