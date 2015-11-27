#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE  8

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

using namespace std;

unsigned char *image_s = NULL;     // source image array
unsigned char *image_t = NULL;     // target image array
FILE *fp_s = NULL;                 // source file handler
FILE *fp_t = NULL;                 // target file handler

unsigned int   width, height;      // image width, image height
unsigned int   rgb_raw_data_offset;// rgb raw data offset
unsigned char  bit_per_pixel;      // bit per pixel
unsigned short byte_per_pixel;     // byte per pixel



// bitmap header
unsigned char header[54] = {
	0x42,        // identity : B
	0x4d,        // identity : M
	0, 0, 0, 0,  // file size
	0, 0,        // reserved1
	0, 0,        // reserved2
	54, 0, 0, 0, // RGB data offset
	40, 0, 0, 0, // struct BITMAPINFOHEADER size
	0, 0, 0, 0,  // bmp width
	0, 0, 0, 0,  // bmp height
	1, 0,        // planes
	24, 0,       // bit per pixel
	0, 0, 0, 0,  // compression
	0, 0, 0, 0,  // data size
	0, 0, 0, 0,  // h resolution
	0, 0, 0, 0,  // v resolution 
	0, 0, 0, 0,  // used colors
	0, 0, 0, 0   // important colors
};

// sobel mask (5x5 version)
// Task 2: Put mask[][][] into Shared Memroy
int _mask[MASK_N][MASK_X][MASK_Y] = {
	{{ -1, -4, -6, -4, -1},
	 { -2, -8,-12, -8, -2},
	 {  0,  0,  0,  0,  0},
	 {  2,  8, 12,  8,  2},
	 {  1,  4,  6,  4,  1}}
,
	{{ -1, -2,  0,  2,  1},
	 { -4, -8,  0,  8,  4},
	 { -6,-12,  0, 12,  6},
	 { -4, -8,  0,  8,  4},
	 { -1, -2,  0,  2,  1}}
};


int read_bmp (const char *fname_s) {
	fp_s = fopen(fname_s, "rb");
	if (fp_s == NULL) {
		printf("fopen fp_s error\n");
		return -1;
	}

	// move offset to 10 to find rgb raw data offset
	fseek(fp_s, 10, SEEK_SET);
	fread(&rgb_raw_data_offset, sizeof(unsigned int), 1, fp_s);

	// move offset to 18 to get width & height;
	fseek(fp_s, 18, SEEK_SET); 
	fread(&width,  sizeof(unsigned int), 1, fp_s);
	fread(&height, sizeof(unsigned int), 1, fp_s);

	// get bit per pixel
	fseek(fp_s, 28, SEEK_SET); 
	fread(&bit_per_pixel, sizeof(unsigned short), 1, fp_s);
	byte_per_pixel = bit_per_pixel / 8;

	// move offset to rgb_raw_data_offset to get RGB raw data
	fseek(fp_s, rgb_raw_data_offset, SEEK_SET);

	// Task 3: Assign image_s to Pinnned Memory
	// Hint  : err = cudaMallocHost ( ... )
	//         if (err != CUDA_SUCCESS)
    int totalSize = width * height * byte_per_pixel;
    totalSize = totalSize < 0 ? -totalSize : totalSize;
	//image_s = (unsigned char *) malloc((size_t)totalSize);
    
    cudaError_t err = cudaMallocHost(&image_s, (size_t)totalSize);
       cudaCheckErrors("cuda_malloc_images_s error"); 
/* 
    if (err != CUDA_SUCCESS) {
       cudaCheckErrors("cuda_malloc_images_s error"); 
    }
*/
/*
    if (image_s == NULL) {
		printf("malloc images_s errori, %d\n", totalSize);
		return -1;
	}
*/
	// Task 3: Assign image_t to Pinned Memory
	// Hint  : err = cudaMallocHost ( ... )
	//         if (err != CUDA_SUCCESS)
	//image_t = (unsigned char *) malloc(totalSize);
    err = cudaMallocHost(&image_t, (size_t)totalSize);
    cudaCheckErrors("cuda_malloc_images_t error"); 
/*
    if (err != CUDA_SUCCESS) {
       cudaCheckErrors("cuda_malloc_images_t error"); 
    }
*/
/*	
    if (image_t == NULL) {
		printf("malloc image_t error %d\n", totalSize);
		return -1;
	}
*/
	fread(image_s, sizeof(unsigned char), (size_t)(long) totalSize, fp_s);

	return 0;
}

typedef unsigned char uint8;
typedef unsigned int uint32;
typedef unsigned short uint16;
// unsigned char *image_s = NULL;     // source image array
// unsigned char *image_t = NULL;     // target image array
// FILE *fp_s = NULL;                 // source file handler
//FILE *fp_t = NULL;                 // target file handler

// unsigned int   width, height;      // image width, image height
// unsigned int   rgb_raw_data_offset;// rgb raw data offset
// unsigned char  bit_per_pixel;      // bit per pixel
// unsigned short byte_per_pixel;     // byte per pixel

void DisplayHeader()
{
    const int kb = 1024;
    const int mb = kb * kb;
    wcout << "NBody.GPU" << endl << "=========" << endl << endl;

    //wcout << "CUDA version:   v" << CUDART_VERSION << endl;    
    //wcout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl << endl; 

    int devCount;
    cudaGetDeviceCount(&devCount);
    wcout << "CUDA Devices: " << endl << endl;

    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        wcout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
        wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
        wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
        wcout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
        wcout << "  Block registers: " << props.regsPerBlock << endl << endl;

        wcout << "  Warp size:         " << props.warpSize << endl;
        wcout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
        wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << endl;
        wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << endl;
        wcout << endl;
    }
}

__global__ void sobel(unsigned char *c_image_s, unsigned char *c_image_t, 
                        unsigned int *c_width, unsigned int *c_height,
                        unsigned short *c_byte_per_pixel,
                        int* c_mask ) {
	int  x, y, i, v, u;            // for loop counter
	int  R, G, B;                  // color of R, G, B
	double val[MASK_N*3] = {0.0};
	int adjustX, adjustY, xBound, yBound;
    unsigned char  *image_s = c_image_s;
    unsigned char  *image_t = c_image_t;
    int width = *c_width;
    int height = *c_height;    
    unsigned short byte_per_pixel = *c_byte_per_pixel;
    __shared__ int mask[MASK_N][MASK_X][MASK_Y];
    
     
    for (int i = 0; i < MASK_N; ++i) {
        for (int j = 0; j < MASK_X; ++j) {
            for(int k = 0; k < MASK_Y; ++k) {
                mask[i][j][k] = 
                    c_mask[i*MASK_Y*MASK_X + j*MASK_Y + k];
            }
        }
    }

    __syncthreads();
	// Task 2: Put mask[][][] into Shared Memory
	// Hint  : Please declare it in kernel function
	//         Then use some threads to move data from global memory to shared memory
	//         Remember to __syncthreads() after it's done <WHY?>

	// Task 1: Relabel x, y into combination of blockIdx, threadIdx ... etc
	// Hint A: We do not have enough threads for each pixels in the image, so what should we do?
	// Hint B: Maybe you can map each y to different threads in different blocks
    int threadNum = blockDim.x;
    
    int blockPerHeight = (width / threadNum) + 
                        ((width % threadNum) > 0 ? 1 : 0) ;

    y = blockIdx.x / blockPerHeight;
    x = threadNum * (blockIdx.x % blockPerHeight) + threadIdx.x;
    	
    if (y < height) {
		if (x < width) {
			for (i = 0; i < MASK_N; ++i) {
				adjustX = (MASK_X % 2) ? 1 : 0;
				adjustY = (MASK_Y % 2) ? 1 : 0;
				xBound = MASK_X /2;
				yBound = MASK_Y /2;

				val[i*3+2] = 0.0;
				val[i*3+1] = 0.0;
				val[i*3] = 0.0;

				for (v = -yBound; v < yBound + adjustY; ++v) {
					for (u = -xBound; u < xBound + adjustX; ++u) {
						if ((x + u) >= 0 && (x + u) < width && y + v >= 0 && y + v < height) {
							R = image_s[byte_per_pixel * (width * (y+v) + (x+u)) + 2];
							G = image_s[byte_per_pixel * (width * (y+v) + (x+u)) + 1];
							B = image_s[byte_per_pixel * (width * (y+v) + (x+u)) + 0];
							val[i*3+2] += R * mask[i][u + xBound][v + yBound];
							val[i*3+1] += G * mask[i][u + xBound][v + yBound];
							val[i*3+0] += B * mask[i][u + xBound][v + yBound];
						}	
					}
				}
			}

			double totalR = 0.0;
			double totalG = 0.0;
			double totalB = 0.0;
			for (i = 0; i < MASK_N; ++i) {
				totalR += val[i*3+2] * val[i*3+2];
				totalG += val[i*3+1] * val[i*3+1];
				totalB += val[i*3+0] * val[i*3+0];
			}

			totalR = sqrt(totalR) / SCALE;
			totalG = sqrt(totalG) / SCALE;
			totalB = sqrt(totalB) / SCALE;
			const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
			const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
			const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
			image_t[ byte_per_pixel * (width * y + x) + 2 ] = cR;
			image_t[ byte_per_pixel * (width * y + x) + 1 ] = cG;
			image_t[ byte_per_pixel * (width * y + x) + 0 ] = cB;
		}
	}
}

int write_bmp (const char *fname_t) {
	unsigned int file_size; // file size

	fp_t = fopen(fname_t, "wb");
	if (fp_t == NULL) {
		printf("fopen fname_t error\n");
		return -1;
	}

	// file size  
	file_size = width * height * byte_per_pixel + rgb_raw_data_offset;
	header[2] = (unsigned char)(file_size & 0x000000ff);
	header[3] = (file_size >> 8)  & 0x000000ff;
	header[4] = (file_size >> 16) & 0x000000ff;
	header[5] = (file_size >> 24) & 0x000000ff;

	// width
	header[18] = width & 0x000000ff;
	header[19] = (width >> 8)  & 0x000000ff;
	header[20] = (width >> 16) & 0x000000ff;
	header[21] = (width >> 24) & 0x000000ff;

	// height
	header[22] = height &0x000000ff;
	header[23] = (height >> 8)  & 0x000000ff;
	header[24] = (height >> 16) & 0x000000ff;
	header[25] = (height >> 24) & 0x000000ff;

	// bit per pixel
	header[28] = bit_per_pixel;

	// write header
	fwrite(header, sizeof(unsigned char), rgb_raw_data_offset, fp_t);

	// write image
	fwrite(image_t, sizeof(unsigned char), (size_t)(long)width * height * byte_per_pixel, fp_t);

	fclose(fp_s);
	fclose(fp_t);

	return 0;
}

int init_device ()
{	// Task 1: Device (GPU) Initialization
	// Hint  : cudaSetDevice()
    cudaSetDevice(1);
	return 0;
}

int
main(int argc, char **argv) {
	init_device();
	DisplayHeader();
	
	

	const char *input = "candy.bmp";
	if (argc > 1) input = argv[1];
	read_bmp(input); // 24 bit gray level image
        
    
    unsigned char *c_image_s = NULL;     // source image array
    unsigned char *c_image_t = NULL;     // target image array

    unsigned int   *c_width, *c_height;      // image width, image height
    unsigned int   *c_rgb_raw_data_offset;// rgb raw data offset
    unsigned char  *c_bit_per_pixel;      // bit per pixel
    unsigned short *c_byte_per_pixel;     // byte per pixel
        
    int *c_mask = NULL;    

	// Task 1: Allocate memory on GPU
	// Hint  : cudaMalloc ()
	//         What do we need to store on GPU? (input image, output image, ...

    int mask1D[MASK_N * MASK_Y * MASK_X];
    for (int i = 0; i < MASK_N; ++i)
        for (int j = 0; j < MASK_X; ++j)
            for(int k = 0; k < MASK_Y; ++k)
                mask1D[i * MASK_X * MASK_Y + j * MASK_Y + k] = _mask[i][j][k];
    cudaMalloc((void**)&c_image_t, (size_t)width * height * byte_per_pixel);
    cudaMalloc((void**)&c_image_s, (size_t)width * height * byte_per_pixel);
	cudaMalloc((void**)&c_width, (size_t)sizeof(int));
	cudaMalloc((void**)&c_height, (size_t)sizeof(int));
	cudaMalloc((void**)&c_rgb_raw_data_offset, (size_t)sizeof(int));
    cudaMalloc((void**)&c_bit_per_pixel, (size_t)sizeof(char));
    cudaMalloc((void**)&c_byte_per_pixel, (size_t)sizeof(short));
    
    cudaMalloc((void**)&c_mask, (size_t)sizeof(int) * MASK_N * MASK_Y * MASK_X);
    cudaCheckErrors("cudamalloc fail");    
/*
    for (int i = 0; i < MASK_N; ++i)
        cudaMalloc((void**)c_mask[i], sizeof(int*) * MASK_X);
*/
    cudaMemcpy(c_mask, mask1D, sizeof(mask1D), cudaMemcpyHostToDevice);

    // Task 1: Memory copy from Host to Device (GPU)
	// Hint  : cudaMemcpy ( ... , cudaMemcpyHostToDevice )

    cudaMemcpy(c_width, &width, sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(c_height, &height, sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(c_image_t, image_t, width * height * byte_per_pixel, cudaMemcpyHostToDevice);
    cudaMemcpy(c_image_s, image_s, width * height * byte_per_pixel, cudaMemcpyHostToDevice);
    cudaMemcpy(c_bit_per_pixel, &bit_per_pixel, sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(c_byte_per_pixel, &byte_per_pixel, sizeof(short), cudaMemcpyHostToDevice);
    cudaCheckErrors("cuda memcpy fail");
	// Hint  : sobel_Kernel <<< ??? , ??? >>> ( ??? );
    
    int blockNum = (width / 1024) + ((width % 1024) ? 1 : 0); 
    wcout << blockNum << endl; 
    blockNum = blockNum * height;  
    
    wcout << blockNum << endl; 
	sobel<<<blockNum, 1024, (size_t)sizeof(int) * MASK_N * MASK_Y * MASK_X>>>( c_image_s, c_image_t, 
                    c_width, c_height, 
                    c_byte_per_pixel,
                    c_mask);

	// Task 1: Memory Copy from Device (GPU) to Host
	// Hint  : cudaMemcpy ( ... , cudaMemcpyDeviceToHost )

	// Task 1: Free memory on device
	// Hint  : cudaFree ( ... )
    cudaMemcpy(image_t, c_image_t, (size_t)width * height * byte_per_pixel, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cuda memcpy back to host fail");
    
    cudaFree(c_image_t);
    cudaFree(c_image_s);
    cudaFree(c_width);
    cudaFree(c_height);
    cudaFree(c_rgb_raw_data_offset);
    cudaFree(c_bit_per_pixel);
    cudaFree(c_byte_per_pixel);

    write_bmp("out.bmp");

	// Task 3: Free Pinned memory
	// Hint  : replace free ( ... ) by cudaFreeHost ( ... )
	cudaFreeHost(image_s);
	cudaFreeHost(image_t);
}
