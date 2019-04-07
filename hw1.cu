#include <stdio.h>
#include <sys/time.h>

#define IMG_DIMENSION 32
#define N_IMG_PAIRS 10000

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

/* we won't load actual files. just fill the images with random bytes */
void load_image_pairs(uchar *images1, uchar *images2) {
    srand(0);
    for (int i = 0; i < N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION; i++) {
        images1[i] = rand() % 256;
        images2[i] = rand() % 256;
    }
}

bool is_in_image_bounds(int i, int j) {
    return (i >= 0) && (i < IMG_DIMENSION) && (j >= 0) && (j < IMG_DIMENSION);
}

uchar local_binary_pattern(uchar *image, int i, int j) {
    uchar center = image[i * IMG_DIMENSION + j];
    uchar pattern = 0;
    if (is_in_image_bounds(i - 1, j - 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j - 1)] >= center) << 7;
    if (is_in_image_bounds(i - 1, j    )) pattern |= (image[(i - 1) * IMG_DIMENSION + (j    )] >= center) << 6;
    if (is_in_image_bounds(i - 1, j + 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j + 1)] >= center) << 5;
    if (is_in_image_bounds(i    , j + 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j + 1)] >= center) << 4;
    if (is_in_image_bounds(i + 1, j + 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j + 1)] >= center) << 3;
    if (is_in_image_bounds(i + 1, j    )) pattern |= (image[(i + 1) * IMG_DIMENSION + (j    )] >= center) << 2;
    if (is_in_image_bounds(i + 1, j - 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j - 1)] >= center) << 1;
    if (is_in_image_bounds(i    , j - 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j - 1)] >= center) << 0;
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
            distance += ((double)SQR(h1[i] - h2[i])) / (h1[i] + h2[i]);
        }
    }
    return distance;
}

/* Your __device__ functions and __global__ kernels here */


__device__ bool is_in_image_bounds_d(int i,int j){
	return ( i>=0) && (i<IMG_DIMENSION) && (j>=0) && (j<IMG_DIMENSION);
}

__device__ uchar d_local_binary_pattern(uchar* image){
	int i= threadIdx.x; 
	int j= threadIdx.y;
	uchar center= image[i*IMG_DIMENSION +j ];
	uchar pattern =0;
	if (is_in_image_bounds_d(i - 1, j - 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j - 1)] >= center) << 7;
 	if (is_in_image_bounds_d(i - 1, j    )) pattern |= (image[(i - 1) * IMG_DIMENSION + (j    )] >= center) << 6;
	if (is_in_image_bounds_d(i - 1, j + 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j + 1)] >= center) << 5;
	if (is_in_image_bounds_d(i    , j + 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j + 1)] >= center) << 4;
	if (is_in_image_bounds_d(i + 1, j + 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j + 1)] >= center) << 3;
	if (is_in_image_bounds_d(i + 1, j    )) pattern |= (image[(i + 1) * IMG_DIMENSION + (j    )] >= center) << 2; 
	if (is_in_image_bounds_d(i + 1, j - 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j - 1)] >= center) << 1;
	if (is_in_image_bounds_d(i    , j - 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j - 1)] >= center) << 0;
 	return pattern;

}
__global__ void image_to_histogram_simple(uchar *image1, OUT int *hist1) {
	uchar pattern=d_local_binary_pattern(image1);
	//we are adding to global memory, which is shared between all the threads in the 
	//system, which may have same pattern so we need AtomicAdd.	
	atomicAdd(&hist1[pattern],1);

}

//load img1 to AHARED MEM . compute the binary patters
//compute the histogram in shared memory
//load it back to gloabl (hist1)
__global__ void image_to_histogram_shared(uchar *image, OUT int *hist1){
	__shared__ uchar shared_image[IMG_DIMENSION];
	__shared__ int shared_hist[256];
	int i= threadIdx.x;
    int j= threadIdx.y;	
	//load img to shared memory
	int k=i*IMG_DIMENSION+j;
	shared_image[k]= image[k]; 
	if(k>256){
		shared_hist[k]=0;
	}
	__syncthreads(); // all threads load "the image"
	uchar center= image[k ];
    uchar pattern =0;
    if (is_in_image_bounds_d(i - 1, j - 1)) pattern |= (shared_image[(i - 1) * IMG_DIMENSION + (j - 1)] >= center) << 7;
    if (is_in_image_bounds_d(i - 1, j    )) pattern |= (shared_image[(i - 1) * IMG_DIMENSION + (j    )] >= center) << 6;
    if (is_in_image_bounds_d(i - 1, j + 1)) pattern |= (shared_image[(i - 1) * IMG_DIMENSION + (j + 1)] >= center) << 5;
    if (is_in_image_bounds_d(i    , j + 1)) pattern |= (shared_image[(i    ) * IMG_DIMENSION + (j + 1)] >= center) << 4;
    if (is_in_image_bounds_d(i + 1, j + 1)) pattern |= (shared_image[(i + 1) * IMG_DIMENSION + (j + 1)] >= center) << 3;
    if (is_in_image_bounds_d(i + 1, j    )) pattern |= (shared_image[(i + 1) * IMG_DIMENSION + (j    )] >= center) << 2;
    if (is_in_image_bounds_d(i + 1, j - 1)) pattern |= (shared_image[(i + 1) * IMG_DIMENSION + (j - 1)] >= center) << 1;
    if (is_in_image_bounds_d(i    , j - 1)) pattern |= (shared_image[(i    ) * IMG_DIMENSION + (j - 1)] >= center) << 0;
	//may to ooptimize the atomicAdd. 
	atomicAdd((&shared_hist[pattern]),1);
	//we need to wait for all the threads to update the "pattern"
	__syncthreads();
	//load from shared mem to global
	if(k<256){
		hist1[k]=shared_hist[k];
	}
}

__global__ void histogram_distance_simple(int *h1, int *h2, OUT double *distance) {
	int i=threadIdx.x;
	distance[i]=0;
	if(h1[i] + h2[i] != 0){
	 distance[i] += ((double)SQR(h1[i] - h2[i])) / (h1[i] + h2[i]);
	}
	__syncthreads();
	int half_length=256/2;
	while(half_length>0){
		if(i<half_length){
			distance[i]=distance[i]+distance[i+half_length];
		}
	__syncthreads(); //note not inside if, otherwise you will have a deadlock	
	half_length /= 2;	
	}

}

int main() {
    uchar *images1; /* we concatenate all images in one huge array */
    uchar *images2;
    CUDA_CHECK( cudaHostAlloc(&images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
    CUDA_CHECK( cudaHostAlloc(&images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );

    load_image_pairs(images1, images2);
    double t_start, t_finish;
    double total_distance;

    /* using CPU */
    printf("\n=== CPU ===\n");
    int histogram1[256];
    int histogram2[256];
    t_start  = get_time_msec();
    for (int i = 0; i < N_IMG_PAIRS; i++) {
        image_to_histogram(&images1[i * IMG_DIMENSION * IMG_DIMENSION], histogram1);
        image_to_histogram(&images2[i * IMG_DIMENSION * IMG_DIMENSION], histogram2);
        total_distance += histogram_distance(histogram1, histogram2);
    }
    t_finish = get_time_msec();
    printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
    printf("total time %f [msec]\n", t_finish - t_start);
	/*----------------------------------------------------------------------------*/
    /* using GPU task-serial */
	/*----------------------------------------------------------------------------*/
    total_distance=0;
	printf("\n=== GPU Task Serial ===\n");
    do { /* do {} while (0): to keep variables inside this block in their own scope. remove if you prefer otherwise */
        /* Your Code Here */
        uchar *gpu_image1, *gpu_image2; 
        int *gpu_hist1, *gpu_hist2; 
        double *gpu_hist_distance; 
        double cpu_hist_distance;

//cudaMalloc to allocate  in global memory, all the threads see it.
		CUDA_CHECK(cudaMalloc((void**)&gpu_image1,sizeof(uchar) * IMG_DIMENSION * IMG_DIMENSION));
		CUDA_CHECK(cudaMalloc((void**)&gpu_image2,sizeof(uchar) * IMG_DIMENSION * IMG_DIMENSION)); 
		CUDA_CHECK(cudaMalloc((void**)&gpu_hist1,sizeof(int)*256)); 
    	CUDA_CHECK(cudaMalloc((void**)&gpu_hist2,sizeof(int)*256)); 
		CUDA_CHECK(cudaMalloc((void**)&gpu_hist_distance,sizeof(double)*256)); 
//threads in one threadblock, we use dim3 to use x and y block id
		dim3 threadsIntb(IMG_DIMENSION,IMG_DIMENSION);
        t_start = get_time_msec();
        for (int i = 0; i < N_IMG_PAIRS; i++) {
//cudaMemcpy to copy from host to global memory
			CUDA_CHECK(cudaMemcpy(gpu_image1,&(images1[i*IMG_DIMENSION*IMG_DIMENSION]),IMG_DIMENSION*IMG_DIMENSION,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(gpu_image2,&(images2[i*IMG_DIMENSION*IMG_DIMENSION]),IMG_DIMENSION*IMG_DIMENSION,cudaMemcpyHostToDevice));
//set to zeor0
			CUDA_CHECK(cudaMemset(gpu_hist1,0,sizeof(int)*256));
			CUDA_CHECK(cudaMemset(gpu_hist2,0,sizeof(int)*256));
			CUDA_CHECK(cudaMemset(gpu_hist_distance,0,sizeof(double)*256));
//call kernels threadsIntb=1024
			image_to_histogram_simple<<<1, threadsIntb>>>(gpu_image1, gpu_hist1);
            image_to_histogram_simple<<<1, threadsIntb>>>(gpu_image2, gpu_hist2);
            histogram_distance_simple<<<1, 256>>>(gpu_hist1, gpu_hist2, gpu_hist_distance);
//cudaMemcpy to copy from gpu to host            
			CUDA_CHECK(cudaMemcpy(&cpu_hist_distance,gpu_hist_distance,sizeof(double),cudaMemcpyDeviceToHost));
            total_distance += cpu_hist_distance;
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        t_finish = get_time_msec();
//free the global memory
		CUDA_CHECK(cudaFree(gpu_image1));
		CUDA_CHECK(cudaFree(gpu_image2));
		CUDA_CHECK(cudaFree(gpu_hist_distance));
		CUDA_CHECK(cudaFree(gpu_hist1));
		CUDA_CHECK(cudaFree(gpu_hist2));

        printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
        printf("total time %f [msec]\n", t_finish - t_start);
    } while (0);
	/*----------------------------------------------------------------------------*/
    /* using GPU task-serial + images and histograms in shared memory */
	/*----------------------------------------------------------------------------*/
    total_distance=0;
	printf("\n=== GPU Task Serial with shared memory ===\n");
	do { /* do {} while (0): to keep variables inside this block in their own scope. remove if you prefer otherwise */
        /* Your Code Here */
        uchar *gpu_image1, *gpu_image2;
        int *gpu_hist1, *gpu_hist2;
        double *gpu_hist_distance;
        double cpu_hist_distance;

//cudaMalloc to allocate  in global memory, all the threads see it.
        CUDA_CHECK(cudaMalloc((void**)&gpu_image1,sizeof(uchar) * IMG_DIMENSION * IMG_DIMENSION));
        CUDA_CHECK(cudaMalloc((void**)&gpu_image2,sizeof(uchar) * IMG_DIMENSION * IMG_DIMENSION));
        CUDA_CHECK(cudaMalloc((void**)&gpu_hist1,sizeof(int)*256));
        CUDA_CHECK(cudaMalloc((void**)&gpu_hist2,sizeof(int)*256));
        CUDA_CHECK(cudaMalloc((void**)&gpu_hist_distance,sizeof(double)*256));
//threads in one threadblock, we use dim3 to use x and y block id
        dim3 threadsIntb(IMG_DIMENSION,IMG_DIMENSION);
	
		t_start = get_time_msec();
		for (int i = 0; i < N_IMG_PAIRS; i++) {
			CUDA_CHECK(cudaMemcpy(gpu_image1,&(images1[i*IMG_DIMENSION*IMG_DIMENSION]),IMG_DIMENSION*IMG_DIMENSION,cudaMemcpyHostToDevice));
    		CUDA_CHECK(cudaMemcpy(gpu_image2,&(images2[i*IMG_DIMENSION*IMG_DIMENSION]),IMG_DIMENSION*IMG_DIMENSION,cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemset(gpu_hist1,0,sizeof(int)*256));
			CUDA_CHECK(cudaMemset(gpu_hist2,0,sizeof(int)*256));
    		CUDA_CHECK(cudaMemset(gpu_hist_distance,0,sizeof(double)*256));
			//shared memory in kernel image_to_histogram_shared
			image_to_histogram_shared<<<1, threadsIntb>>>(gpu_image1, gpu_hist1);
    		image_to_histogram_shared<<<1, threadsIntb>>>(gpu_image2, gpu_hist2);
    		histogram_distance_simple<<<1, 256>>>(gpu_hist1, gpu_hist2, gpu_hist_distance);
     		CUDA_CHECK(cudaMemcpy(&cpu_hist_distance,gpu_hist_distance,sizeof(double),cudaMemcpyDeviceToHost));
			total_distance += cpu_hist_distance;
    	}
    	CUDA_CHECK(cudaDeviceSynchronize());
		t_finish = get_time_msec();
    	printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
    	printf("total time %f [msec]\n", t_finish - t_start);
	}while(0);
	/*----------------------------------------------------------------------------*/
    /* using GPU + batching */
	/*----------------------------------------------------------------------------*/
	total_distance=0;
	printf("\n=== GPU Batching ===\n");
    /* Your Code Here */
    printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
    printf("total time %f [msec]\n", t_finish - t_start);

    return 0;
}
