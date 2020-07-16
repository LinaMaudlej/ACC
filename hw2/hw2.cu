
		#include <stdio.h>
		#include <sys/time.h>
		#include <unistd.h>
		#include <assert.h>
		#include <string.h>

		#define IMG_DIMENSION 32
		#define N_IMG_PAIRS 10000
		#define NREQUESTS 10000

		#define SLOTSNUM 10
		

		typedef unsigned char uchar;
		#define OUT

		#define CUDA_CHECK(f) do {                                                                  \
			cudaError_t e = f;                                                                      \
			if (e != cudaSuccess) {                                                                 \
				printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
				exit(1);                                                                            \
			}                                                                                       \
		} while (0)

		#define SQR(a) ((a) * (a))
		
		
	

		double static inline get_time_msec(void) {
				struct timeval t;
				gettimeofday(&t, NULL);
				return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
		}

		/* we'll use these to rate limit the request load */
		struct rate_limit_t {
				double last_checked;
				double lambda;
				unsigned seed;
		};

		void rate_limit_init(struct rate_limit_t *rate_limit, double lambda, int seed) {
				rate_limit->lambda = lambda;
				rate_limit->seed = (seed == -1) ? 0 : seed;
				rate_limit->last_checked = 0;
		}

		int rate_limit_can_send(struct rate_limit_t *rate_limit) {
				if (rate_limit->lambda == 0)
						return 1;
				double now = get_time_msec() * 1e-3;
				double dt = now - rate_limit->last_checked;
				double p = dt * rate_limit->lambda;
				rate_limit->last_checked = now;
				if (p > 1)
						p = 1;
				double r = (double) rand_r(&rate_limit->seed) / RAND_MAX;
				return (p > r);
		}

		void rate_limit_wait(struct rate_limit_t *rate_limit) {
				while (!rate_limit_can_send(rate_limit)) {
						usleep(1. / (rate_limit->lambda * 1e-6) * 0.01);
				}
		}

		/* we won't load actual files. just fill the images with random bytes */
		void load_image_pairs(uchar *images1, uchar *images2) {
				srand(0);
				for (int i = 0; i < N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION; i++) {
						images1[i] = rand() % 256;
						images2[i] = rand() % 256;
				}
		}

		__device__ __host__ bool is_in_image_bounds(int i, int j) {
				return (i >= 0) && (i < IMG_DIMENSION) && (j >= 0) && (j < IMG_DIMENSION);
		}

		__device__         __host__ uchar local_binary_pattern(uchar *image, int i, int j) {
				uchar center = image[i * IMG_DIMENSION + j];
				uchar pattern = 0;
				if (is_in_image_bounds(i - 1, j - 1))
						pattern |= (image[(i - 1) * IMG_DIMENSION + (j - 1)] >= center) << 7;
				if (is_in_image_bounds(i - 1, j))
						pattern |= (image[(i - 1) * IMG_DIMENSION + (j)] >= center) << 6;
				if (is_in_image_bounds(i - 1, j + 1))
						pattern |= (image[(i - 1) * IMG_DIMENSION + (j + 1)] >= center) << 5;
				if (is_in_image_bounds(i, j + 1))
						pattern |= (image[(i) * IMG_DIMENSION + (j + 1)] >= center) << 4;
				if (is_in_image_bounds(i + 1, j + 1))
						pattern |= (image[(i + 1) * IMG_DIMENSION + (j + 1)] >= center) << 3;
				if (is_in_image_bounds(i + 1, j))
						pattern |= (image[(i + 1) * IMG_DIMENSION + (j)] >= center) << 2;
				if (is_in_image_bounds(i + 1, j - 1))
						pattern |= (image[(i + 1) * IMG_DIMENSION + (j - 1)] >= center) << 1;
				if (is_in_image_bounds(i, j - 1))
						pattern |= (image[(i) * IMG_DIMENSION + (j - 1)] >= center) << 0;
				return pattern;
		}

		void image_to_histogram(uchar *image, int *histogram) {
				memset(histogram, 0, sizeof(int) * 256);
				for (int i = 0; i < IMG_DIMENSION; i++) {
						for (int j = 0; j < IMG_DIMENSION; j++) {
								uchar pattern = local_binary_pattern(image, i, j);
								histogram[pattern]++;
						}
				}
		}

		double histogram_distance(int *h1, int *h2) {
				/* we'll use the chi-square distance */
				double distance = 0;
				for (int i = 0; i < 256; i++) {
						if (h1[i] + h2[i] != 0) {
								distance += ((double) SQR(h1[i] - h2[i])) / (h1[i] + h2[i]);
						}
				}
				return distance;
		}

		__global__ void gpu_image_to_histogram(uchar *image, int *histogram) {
				uchar pattern = local_binary_pattern(image, threadIdx.x / IMG_DIMENSION,
								threadIdx.x % IMG_DIMENSION);
				atomicAdd(&histogram[pattern], 1);
		}

		__global__ void gpu_histogram_distance(int *h1, int *h2, double *distance) {
				int length = 256;
				int tid = threadIdx.x;
				distance[tid] = 0;
				if (h1[tid] + h2[tid] != 0) {
						distance[tid] = ((double) SQR(h1[tid] - h2[tid])) / (h1[tid] + h2[tid]);
				}
				__syncthreads();

				while (length > 1) {
						if (threadIdx.x < length / 2) {
								distance[tid] = distance[tid] + distance[tid + length / 2];
						}
						length /= 2;
						__syncthreads();
				}
		}

		void print_usage_and_die(char *progname) {
				printf("usage:\n");
				printf("%s streams <load (requests/sec)>\n", progname);
				printf("OR\n");
				printf("%s queue <#threads> <load (requests/sec)>\n", progname);
				exit(1);
		}

		typedef struct requset {
				int num; 
				double distance;
				uchar *images1;
				uchar *images2;
		} Request;

		typedef struct queue {
				int tail;
				int head;
				Request requestsQueue[SLOTSNUM];
		}*Queue;

		///////////////////////////////////////////////////////////////////////////
		////////////////////////// Host FUNCTION ///////////////////////////////////

		__host__ void queueInit(Queue queue) {
				queue->head = 0; //strating form 0
				queue->tail = SLOTSNUM - 1; //starting from 9
		}

		__host__ bool isResponseExists(Queue response_queue, Request* response) {
				int tmp=(response_queue->tail + 1) % SLOTSNUM;
				__sync_synchronize();

				if (tmp == response_queue->head) {
						return false;
				}
				*response = response_queue->requestsQueue[tmp];
				response_queue->tail = tmp;
				__sync_synchronize();
				return true;
		}

		__host__ bool isRequestInserted(Queue request_queue, Request newRequest, cudaStream_t* copyingStream) {
				int tmp=(request_queue->head + 1) % SLOTSNUM;
				__sync_synchronize();
				if ( tmp== request_queue->tail) {
						return false;
				}
				request_queue->requestsQueue[request_queue->head] = newRequest;
				__sync_synchronize();
				while (cudaStreamQuery(*copyingStream) != cudaSuccess);
				request_queue->head = tmp;
				__sync_synchronize();
				return true;
		}

		///////////////////////////////////////////////////////////////////////////
		////////////////////////// GPU FUNCTION ///////////////////////////////////
		__device__ void gpu_image_to_histogram_tmp(uchar *image, int *histogram) {
			for (int i = threadIdx.x; i < IMG_DIMENSION * IMG_DIMENSION; i += blockDim.x){
				uchar pattern = local_binary_pattern(image, i / IMG_DIMENSION,
								i % IMG_DIMENSION);
				atomicAdd(&histogram[pattern], 1);
			}
		}

		__device__ void gpu_histogram_distance_tmp(int *h1, int *h2, double *distance) {
				int length = 256;
			//	if(tid<256){
				for (int tid = threadIdx.x; tid < length; tid += blockDim.x) {
					distance[tid] = 0;
					if (h1[tid] + h2[tid] != 0) {
							distance[tid] = ((double) SQR(h1[tid] - h2[tid])) / (h1[tid] + h2[tid]);
					}
				}
				__syncthreads();

				while (length > 1) {
						if (threadIdx.x < length / 2) {
								for (int tid = threadIdx.x; tid < length/2; tid += blockDim.x) {
								distance[tid] = distance[tid] + distance[tid + length / 2];
								}
						}
						length /= 2;
						__syncthreads();
				}
			//	}
		}

		__device__ bool isResponseInserted(Queue response_queue, Request newResponse) {
				int tmp=(response_queue->head + 1) % SLOTSNUM;
				__threadfence_system();
				if (tmp == response_queue->tail) {
						return false;
				}
				response_queue->requestsQueue[response_queue->head] = newResponse;
				__threadfence_system();
				response_queue->head = tmp;
				__threadfence_system();
				return true;
		}

		__device__ bool isRequestExists(Queue request_queue, Request* request) {
				int tmp=(request_queue->tail + 1) % SLOTSNUM;
				__threadfence_system();
				if (tmp == request_queue->head) {
						return false;
				}
				*request = request_queue->requestsQueue[tmp];
				__threadfence_system();
				request_queue->tail = tmp;
				__threadfence_system();
				return true;
		}

		__global__ void server_kernel(Queue request_queue, Queue response_queue,bool* flag) {
			Queue RequestQueue = &request_queue[blockIdx.x];
			Queue ResponseQueue = &response_queue[blockIdx.x];
			 __shared__ Request requestDequeued; //output (dequeue response from queue)
			 __shared__ int hist1[256];
			 __shared__ int hist2[256];
			 __shared__ double distance[256];
				while (*flag==false) { //stop all threadblocks from running (queues are emptry)
						if (threadIdx.x == 0) {//when threadnumber is zero try te dequeue a request
								while (!isRequestExists(RequestQueue, &requestDequeued));
							  for (int i = 0; i < 256; i++) {
									  hist1[i] = 0;
									  hist2[i] = 0;
							  }
						}
						__syncthreads();
						gpu_image_to_histogram_tmp(requestDequeued.images1, hist1);
						gpu_image_to_histogram_tmp(requestDequeued.images2, hist2);
						__syncthreads();
						gpu_histogram_distance_tmp(hist1, hist2, distance);
					   __syncthreads(); //making sure all threads are here
						if (threadIdx.x == 0) { 
							  Request currResponse;
							  currResponse.distance = distance[0];
							  currResponse.num = requestDequeued.num;
							  //currResponse.images1 = requestDequeued.images1;
							  //currResponse.images2 = requestDequeued.images2;
								while (!isResponseInserted(ResponseQueue, currResponse)); ////response is ready, enqueue it
						}
				}
		}
		
		
		///////////////////////////////////////////////////////////////////////////
		////////////////////////// NUM OF BLOCK CAL ///////////////////////////////////
		
		int BLOCKNUM =16;
		
		void changeBlocksNum(int _threadsUsed){
			/*
			The usage of three types of resources affect the number of concurrently scheduled blocks. 
			They are, Threads Per Block, Registers Per Block and Shared Memory Per Block
			for example:start increasing Shared Memory Per Block, the Max Blocks per Multiprocessor will continue 
			to be your limiting factor until you reach a threshold at which Shared Memory Per Block becomes the
			limiting factor.
			*/
			cudaDeviceProp prop;
			CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
			/*
			checking how much we have used
			*/
			struct cudaFuncAttributes func;
			CUDA_CHECK(cudaFuncGetAttributes(&func,server_kernel));
	
			// Registers Per Block
			int maxRegisters=prop.regsPerMultiprocessor;
			int regsUsed=func.numRegs;
			int num2=maxRegisters/regsUsed;
			// Shared Memory Per Block
			int maxSharedMem=prop.sharedMemPerMultiprocessor;
			int sharedMemUsed=func.sharedSizeBytes;
			int num3=maxSharedMem/sharedMemUsed;
			//Threads Per Block
			//if(_threadsUsed<prop.maxThreadsPerBlock){
			int maxThreads=prop.maxThreadsPerMultiProcessor;
			int threadsUsed=prop.maxThreadsPerBlock; //not all of them used!! TODO
			int num1=maxThreads/threadsUsed;
			//}
			//printf("num1 is %d  num2 %d num3 %d\n ",num1,num2,num3);
			int num=min(min(num1,num2),num3); // per multiprocessor
			BLOCKNUM=prop.multiProcessorCount*num;
			//printf("minum is %d   numberBlocks %d\n",num,BLOCKNUM);

		} 

		/////////////////////////////////////////////////////////////////////////////////
		enum {
				PROGRAM_MODE_STREAMS = 0, PROGRAM_MODE_QUEUE
		};
		int main(int argc, char *argv[]) {
				
				int mode = -1;
				int threads_queue_mode = -1; /* valid only when mode = queue */
				double load = 0;
				if (argc < 3)
						print_usage_and_die(argv[0]);

				if (!strcmp(argv[1], "streams")) {
						if (argc != 3)
								print_usage_and_die(argv[0]);
						mode = PROGRAM_MODE_STREAMS;
						load = atof(argv[2]);
				} else if (!strcmp(argv[1], "queue")) {
						if (argc != 4)
								print_usage_and_die(argv[0]);
						mode = PROGRAM_MODE_QUEUE;
						threads_queue_mode = atoi(argv[2]);
				        changeBlocksNum(threads_queue_mode); //added
						load = atof(argv[3]);
				} else {
						print_usage_and_die(argv[0]);
				}
				uchar *images1; /* we concatenate all images in one huge array */
				uchar *images2;
				CUDA_CHECK(cudaHostAlloc(&images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0));
				CUDA_CHECK(cudaHostAlloc(&images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0));

				load_image_pairs(images1, images2);
				double t_start, t_finish;
				double total_distance;

				/* using CPU */
				printf("\n=== CPU ===\n");
				int histogram1[256];
				int histogram2[256];
				t_start = get_time_msec();
				for (int i = 0; i < NREQUESTS; i++) {
						int img_idx = i % N_IMG_PAIRS;
						image_to_histogram(&images1[img_idx * IMG_DIMENSION * IMG_DIMENSION],
										histogram1);
						image_to_histogram(&images2[img_idx * IMG_DIMENSION * IMG_DIMENSION],
										histogram2);
						total_distance += histogram_distance(histogram1, histogram2);
				}
				t_finish = get_time_msec();
				printf("average distance between images %f\n", total_distance / NREQUESTS);
				printf("throughput = %lf (req/sec)\n",
								NREQUESTS / (t_finish - t_start) * 1e+3);

				/* using GPU task-serial.. just to verify the GPU code makes sense */
				printf("\n=== GPU Task Serial ===\n");
				do {
						uchar *gpu_image1, *gpu_image2; // TODO: allocate with cudaMalloc
						int *gpu_hist1, *gpu_hist2; // TODO: allocate with cudaMalloc
						double *gpu_hist_distance; //TODO: allocate with cudaMalloc
						double cpu_hist_distance;
						cudaMalloc(&gpu_image1, IMG_DIMENSION * IMG_DIMENSION);
						cudaMalloc(&gpu_image2, IMG_DIMENSION * IMG_DIMENSION);
						cudaMalloc(&gpu_hist1, 256 * sizeof(int));
						cudaMalloc(&gpu_hist2, 256 * sizeof(int));
						cudaMalloc(&gpu_hist_distance, 256 * sizeof(double));

						total_distance = 0;
						t_start = get_time_msec();
						for (int i = 0; i < NREQUESTS; i++) {
								int img_idx = i % N_IMG_PAIRS;
								cudaMemcpy(gpu_image1,
												&images1[img_idx * IMG_DIMENSION * IMG_DIMENSION],
												IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice);
								cudaMemcpy(gpu_image2,
												&images2[img_idx * IMG_DIMENSION * IMG_DIMENSION],
												IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice);
								cudaMemset(gpu_hist1, 0, 256 * sizeof(int));
								cudaMemset(gpu_hist2, 0, 256 * sizeof(int));
								gpu_image_to_histogram<<<1, 1024>>>(gpu_image1, gpu_hist1);
								gpu_image_to_histogram<<<1, 1024>>>(gpu_image2, gpu_hist2);
								gpu_histogram_distance<<<1, 256>>>(gpu_hist1, gpu_hist2,
												gpu_hist_distance);
								cudaMemcpy(&cpu_hist_distance, gpu_hist_distance, sizeof(double),
												cudaMemcpyDeviceToHost);
								total_distance += cpu_hist_distance;
						}
						CUDA_CHECK(cudaDeviceSynchronize());
						t_finish = get_time_msec();
						printf("average distance between images %f\n",
										total_distance / NREQUESTS);
						printf("throughput = %lf (req/sec)\n",
										NREQUESTS / (t_finish - t_start) * 1e+3);
				} while (0);

				/* now for the client-server part */
				printf("\n=== Client-Server ===\n");
				total_distance = 0;
				double *req_t_start = (double *) malloc(NREQUESTS * sizeof(double));
				memset(req_t_start, 0, NREQUESTS * sizeof(double));

				double *req_t_end = (double *) malloc(NREQUESTS * sizeof(double));
				memset(req_t_end, 0, NREQUESTS * sizeof(double));

				struct rate_limit_t rate_limit;
				rate_limit_init(&rate_limit, load, 0);

				/* TODO allocate / initialize memory, streams, etc... */
				//***********************************************//
				//initialize streams
				cudaStream_t streams[64];
				for (int i = 0; i < 64; i++) {
						cudaStreamCreate(&streams[i]);
				}
				uchar *gpu_image1, *gpu_image2; // TODO: allocate with cudaMalloc
				int *gpu_hist1, *gpu_hist2; // TODO: allocate with cudaMalloc
				double *gpu_hist_distance; //TODO: allocate with cudaMalloc
				double cpu_hist_distance[64]; // ?
				CUDA_CHECK(
								cudaMallocHost((void**)&gpu_image1, IMG_DIMENSION * IMG_DIMENSION)); //cudaHostAlloc??? did not work
				CUDA_CHECK(
								cudaMallocHost((void**)&gpu_image2, IMG_DIMENSION * IMG_DIMENSION));
				CUDA_CHECK(cudaMallocHost((void** )&gpu_hist1, 256 * sizeof(int)));
				CUDA_CHECK(cudaMallocHost((void** )&gpu_hist2, 256 * sizeof(int)));
				CUDA_CHECK(
								cudaMallocHost((void** )&gpu_hist_distance, 256 * sizeof(double)));

				int hash_streamNumToReqNum[64]; //numReq=hash_streamNumToReqNum[0] :stream num 0  with requst number
				for (int i = 0; i < 64; i++) {
						hash_streamNumToReqNum[i] = -1; //-1
				}
				int freeStream = -1; //if it is -1 ,pick up new freeStream
				//***********************************************//

				double ti = get_time_msec();
				if (mode == PROGRAM_MODE_STREAMS) {
						for (int i = 0; i < NREQUESTS; i++) {
								//***********************************************//
								/* TODO query (don't block) streams for any completed requests.
								 update req_t_end of completed requests
								 update total_distance */
								//using cudaStreamQuery which Returns cudaSuccess if all operations in stream have completed
								//pick up free stream
								for (int str = 0; str < 64; str++) {
										if (cudaStreamQuery(streams[str]) == cudaSuccess
														&& hash_streamNumToReqNum[str] != -1) { //finished tasks in the stream num str
												req_t_end[hash_streamNumToReqNum[str]] = get_time_msec();
												total_distance = total_distance + cpu_hist_distance[str];
												cpu_hist_distance[str] = 0;
												freeStream = str; //done, this is now the freeStream
												break;

										} else if (hash_streamNumToReqNum[str] == -1) { //it is a free stream , take it
												freeStream = str;
												break;
										} else if (str == 63) { //waiting for stream to be finished (not blocking)
												str = 0;
												continue;
										}

								}
								hash_streamNumToReqNum[freeStream] = i; //this free stream now has the req i
								//***********************************************//

								rate_limit_wait(&rate_limit);
								req_t_start[i] = get_time_msec();
								int img_idx = i % N_IMG_PAIRS;

								/* TODO place memcpy's and kernels in a stream */
								//***********************************************//
								CUDA_CHECK(
												cudaMemcpyAsync(&gpu_image1[freeStream * IMG_DIMENSION * IMG_DIMENSION], &images1[img_idx * IMG_DIMENSION * IMG_DIMENSION], IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice,streams[freeStream]));
								CUDA_CHECK(
												cudaMemcpyAsync(&gpu_image2[freeStream * IMG_DIMENSION * IMG_DIMENSION], &images2[img_idx * IMG_DIMENSION * IMG_DIMENSION], IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice,streams[freeStream]));
								CUDA_CHECK(
												cudaMemsetAsync(&gpu_hist1[freeStream * 256], 0,
																256 * sizeof(int), streams[freeStream]));
								CUDA_CHECK(
												cudaMemsetAsync(&gpu_hist2[freeStream * 256], 0,
																256 * sizeof(int), streams[freeStream]));
								gpu_image_to_histogram<<<1, 1024, 0, streams[freeStream]>>>(
												gpu_image1, gpu_hist1);
								gpu_image_to_histogram<<<1, 1024, 0, streams[freeStream]>>>(
												gpu_image2, gpu_hist2);
								gpu_histogram_distance<<<1, 256, 0, streams[freeStream]>>>(
												gpu_hist1, gpu_hist2, &gpu_hist_distance[freeStream * 256]);
								CUDA_CHECK(
												cudaMemcpyAsync(&cpu_hist_distance[freeStream],
																&gpu_hist_distance[freeStream * 256],
																sizeof(double), cudaMemcpyDeviceToHost,
																streams[freeStream]));
								//***********************************************//
						}
						/* TODO now make sure to wait for all streams to finish */
						//***********************************************//
						for (int j = 0; j < 64; ++j) {
								if (cudaStreamSynchronize(streams[j]) == cudaSuccess) {
										req_t_end[hash_streamNumToReqNum[j]] = get_time_msec();
										total_distance = total_distance + cpu_hist_distance[j];
								}
						}
						//***********************************************//

				} else if (mode == PROGRAM_MODE_QUEUE) {
						uchar *gpu_image1, *gpu_image2; // TODO: allocate with cudaMalloc
						
						CUDA_CHECK(cudaMalloc(&gpu_image1, NREQUESTS* IMG_DIMENSION * IMG_DIMENSION));
						CUDA_CHECK(cudaMalloc(&gpu_image2, NREQUESTS* IMG_DIMENSION * IMG_DIMENSION));
						cudaStream_t executeStream;
						cudaStream_t copyingStream;
						/*
						we have to queues, queue for requests and queue for respons
						every queue has its own "head" and "tail" which indiacts the encoded/decoded place in queue
						every qeueue has an array for requests or respons 
						which includes msg requested (img1 & img2 ) the request number (for adding calculated time) to the req
						also 
						*/
						Queue request_q_dev; 
						Queue request_q_host;
						Queue response_q_dev;
						Queue response_q_host;
						/*
						resCounters & reqCounters to indicates if there are requests to encoded/decoded
						before killing threadblocks
						*/
						int* resCounters= (int*)malloc(sizeof(int)*BLOCKNUM);//reponses counter
						int* reqCounters= (int*)malloc(sizeof(int)*BLOCKNUM); //requests counter
						for(int k=0;k<BLOCKNUM;k++){
							reqCounters[k]=0;
							resCounters[k]=0;
						}
						
						
						
						CUDA_CHECK(cudaStreamCreate(&executeStream));
						CUDA_CHECK(cudaStreamCreate(&copyingStream));

						CUDA_CHECK(cudaHostAlloc(&request_q_host, BLOCKNUM * sizeof(struct queue),
										cudaHostAllocMapped));
						CUDA_CHECK(cudaHostAlloc(&response_q_host, BLOCKNUM * sizeof(struct queue),
										cudaHostAllocMapped));
						CUDA_CHECK(cudaHostGetDevicePointer(&request_q_dev, request_q_host, 0));
						CUDA_CHECK(cudaHostGetDevicePointer(&response_q_dev, response_q_host, 0));

						//Initilize queues, now requests and no respons
						for (int i = 0; i < BLOCKNUM; i++) {
								queueInit(&request_q_host[i]);
								queueInit(&response_q_host[i]);
						}
						__sync_synchronize();
						/*
						this flag to kill running threadblocks
						*/
						bool* finish_dev;
						bool* finish_host;
						CUDA_CHECK(cudaHostAlloc(&finish_host,  sizeof( bool), cudaHostAllocMapped));
						*finish_host=false;
						CUDA_CHECK(cudaHostGetDevicePointer(&finish_dev, finish_host, 0));
						
						__sync_synchronize(); 

						server_kernel<<<BLOCKNUM, threads_queue_mode, 0, executeStream>>>(
										request_q_dev, response_q_dev,finish_dev);

						Request newRequest;
						int blockToTry = -1;

						for (int i = 0; i < NREQUESTS; i++) {
								/* TODO check producer consumer queue for any responses.
								 don't block. if no responses are there we'll check again in the next iteration
								 update req_t_end of completed requests
								 update total_distance */
								rate_limit_wait(&rate_limit);
								int img_idx = i % N_IMG_PAIRS;
								CUDA_CHECK(cudaMemcpyAsync(gpu_image1+ (i* IMG_DIMENSION * IMG_DIMENSION),
												&images1[ img_idx * IMG_DIMENSION * IMG_DIMENSION],
												IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice,
												copyingStream));
								CUDA_CHECK(cudaMemcpyAsync(gpu_image2+ (i* IMG_DIMENSION * IMG_DIMENSION),
												&images2[img_idx * IMG_DIMENSION * IMG_DIMENSION],
												IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice,
												copyingStream));
								newRequest.num = i;
								newRequest.images1 = gpu_image1+ (i* IMG_DIMENSION * IMG_DIMENSION);
								newRequest.images2 = gpu_image2+ (i* IMG_DIMENSION * IMG_DIMENSION);
								//printf("I am the request Number %d\n",i);
								do {
										//release all finished responses
										for (int j = 0; j < BLOCKNUM; j++) {
												//printf("checking block %d\n",j );
												Request currResponse;
												while (isResponseExists(&response_q_host[j], &currResponse)) {
													//	printf("trying to deqeue req Num %d ,in block number %d\n",currResponse.num,j);
														req_t_end[currResponse.num] = get_time_msec();
														total_distance += currResponse.distance;
														resCounters[j]++;
														
												}
										}
										blockToTry = (blockToTry + 1) % BLOCKNUM;
								} while (isRequestInserted(&request_q_host[blockToTry], newRequest, &copyingStream)
												== 0);
								reqCounters[blockToTry]++;
								req_t_start[i] = get_time_msec();
								/* TODO place memcpy's and kernels in a stream */
						}
						/* TODO wait until you have responses for all requests */
						for (int reqTodo = 0; reqTodo < BLOCKNUM; reqTodo++) {
								while (resCounters[reqTodo] < reqCounters[reqTodo]) { //waiting for respons for all requests
										Request currResponse;
										while (isResponseExists(&response_q_host[reqTodo], &currResponse) != 0) {
												resCounters[reqTodo]++;
												req_t_end[currResponse.num] = get_time_msec();
												total_distance += currResponse.distance;
												
										}
								}
						}
						*finish_host=true;
						CUDA_CHECK(cudaStreamDestroy(executeStream));
						CUDA_CHECK(cudaStreamDestroy(copyingStream));
						//CUDA_CHECK(cudaFree(gpu_image1)); //terminated?? TODO
						//CUDA_CHECK(cudaFree(gpu_image1)); //terminated?? TODO
				} else {
						assert(0);
				}
				double tf = get_time_msec();

				double avg_latency = 0;
				for (int i = 0; i < NREQUESTS; i++) {
						avg_latency += (req_t_end[i] - req_t_start[i]);
				}
				avg_latency /= NREQUESTS;

				printf("mode = %s\n", mode == PROGRAM_MODE_STREAMS ? "streams" : "queue");
				printf("load = %lf (req/sec)\n", load);
				if (mode == PROGRAM_MODE_QUEUE)
						printf("threads = %d\n", threads_queue_mode);
				printf("average distance between images %f\n", total_distance / NREQUESTS);
				printf("throughput = %lf (req/sec)\n", NREQUESTS / (tf - ti) * 1e+3);
				printf("average latency = %lf (msec)\n", avg_latency);
				//****************//added//****************//
				for (int i = 0; i < 64; i++) {
						cudaStreamDestroy(streams[i]);
				}
				//***********************************************//

				return 0;
		}
