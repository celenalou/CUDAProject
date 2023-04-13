#include <stdio.h>


__global__ void thomas_GPU(const float *a, const float *d, const float *c, const float *y, float *x, unsigned int N)
{
	/* Allocate shared memory space. */
	extern __shared__ float c_Sh[];
	//extern __shared__ float x_Sh[];
	if (!c_Sh)
	{	
		printf("Error in allocation c_Sh\n ");
		return;
	}

	c_Sh[0] = c[0] / d[0];
	x[0] = y[0] / d[0];

	/* loop from 1 to N - 1 inclusive */
	for (int i = 1; i < N; i++)
	{
		const float dino = 1.0f / (d[i] - a[i] * c_Sh[i - 1]);
		c_Sh[i] = c[i] * dino;
		x[i] = (y[i] - a[i] * x[i - 1]) * dino;
	}

	/* loop from x - 2 to 0 inclusive, safely testing loop end condition */
	
	for (int i = N - 2; i > 0; i--){
		x[i] -= c_Sh[i] * x[i + 1];

	}
	x[0] -= c_Sh[0] * x[1];

}


__global__ void CR(const float *a, const float *d, const float *c, const float *y, float *x, unsigned int N){
	//forward reduction
	extern __shared__ float s[];
	float *a_Sh = s;
	float *d_Sh = &a_Sh[N];
	float *c_Sh = &d_Sh[N];
	float *y_Sh = &c_Sh[N];
	float *x_Sh = &y_Sh[N];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int nb_Threads = N/2;
	//printf("nb_steps: %d", nb_steps);
	//Copy data to shared memory
	a_Sh[idx] = a[idx];
	a_Sh[idx+N/2] = a[idx+N/2];
	d_Sh[idx] = d[idx];
	d_Sh[idx+N/2] = d[idx+N/2];
	c_Sh[idx] = c[idx];
	c_Sh[idx+N/2] = c[idx+N/2];
	y_Sh[idx] = y[idx];
	y_Sh[idx+N/2] = y[idx+N/2];
	__syncthreads();
	//Forward reduction
	int stride =1;
	int delta, coef;
	//printf("\n %d TREAD %d id", nb_Threads, threadIdx.x);
	while(nb_Threads>1){
		__syncthreads();
		coef = 1;
		stride *= 2;
		delta = stride /2;
		//printf("\n*** Iteration\n\n");
		if (threadIdx.x < nb_Threads){
			int i = stride * threadIdx.x + stride - 1;
			int i_L = i - delta;
			int i_R = i + delta;
			if (i_R > N - 1){
				i_R = N - 1 ;
				coef = 0;
			} 
			float k_1 = a_Sh[i] / d_Sh[i_L];
			float k_2 = c_Sh[i] / d_Sh[i_R];
			d_Sh[i] = d_Sh[i] - c_Sh[i_L] * k_1 - a_Sh[i_R] * coef * k_2;
			y_Sh[i] = y_Sh[i] - y_Sh[i_L] * k_1 - y_Sh[i_R]  * coef * k_2;
			a_Sh[i] = -a_Sh[i_L] * k_1;
			c_Sh[i] = -c_Sh[i_R] * k_2;
			//printf("Forward: %d   L %d   R %d\n", i, i_L, i_R);
			}
		nb_Threads /= 2;
		
	}
	__syncthreads();
	//Backward substitution
	if(threadIdx.x < 2){
		int addr1 = stride - 1 ;
		int addr2 = 2 * stride - 1;
		//printf("Adress: %d %d\n", addr1, addr2);
		float tmp = d_Sh[addr2] * d_Sh[addr1] - c_Sh[addr1] *  a_Sh[addr2];
		x_Sh[addr1] = (d_Sh[addr2] * y_Sh[addr1] - c_Sh[addr1] * y_Sh[addr2]) / tmp;
		x_Sh[addr2] = (y_Sh[addr2] * d_Sh[addr1] - y_Sh[addr1] * a_Sh[addr2]) / tmp;
	}
	nb_Threads = 2;
	while(nb_Threads <= N/2){
		
		delta = stride/2;
		__syncthreads();
		if(threadIdx.x < nb_Threads)
		{
			int i = stride * threadIdx.x + stride/2 - 1; 
			if(i == delta - 1){
				x_Sh[i] = (y_Sh[i] - c_Sh[i] * x_Sh[i + delta]) / d_Sh[i];
				//printf("%d + %d\n", i, i + delta);
			}
			else{
				x_Sh[i] = (y_Sh[i] - a_Sh[i] * x_Sh[i - delta] - c_Sh[i] * x_Sh[i+delta]) / d_Sh[i];
				//printf("%d -%d + %d\n", i, i-delta, i + delta);
			}
		}
		stride /= 2;
		nb_Threads *= 2;
	}
	__syncthreads();
	//copy result from shared memory to x
	x[idx] = x_Sh[idx];
	x[idx+N/2] = x_Sh[idx+N/2];
}


__global__ void PCR(const float *a, const float *d, const float *c, const float *y, float *x, unsigned int N, unsigned int nb_steps)
{
    // seperation of the shared memory into several tab
    extern __shared__ float s[];
    float *a_Sh = s;
    float *d_Sh = &a_Sh[N];
    float *c_Sh = &d_Sh[N];
    float *y_Sh = &c_Sh[N];
    float *x_Sh = &y_Sh[N];

    float aNew, dNew, cNew, yNew;
	int idx = threadIdx.x;

    //Copy data to shared memory

        a_Sh[idx] = a[idx];
        d_Sh[idx] = d[idx];
        c_Sh[idx] = c[idx];
        y_Sh[idx] = y[idx];

    __syncthreads();

    int delta = 1;
    for (int j = 0; j < nb_steps; j++)
    {
        int i = threadIdx.x;
        int i_R = i + delta;
        int coef_L = 1, coef_R=1;
        if (i_R >= N)
        {
            i_R = N - 1;
            coef_R = 0;
        }
        int i_L = i - delta;
        if (i_L < 0)
        {
            i_L = 0;
            coef_L = 0;
        }

        float k1 = a_Sh[i] / d_Sh[i_L];
        float k2 = c_Sh[i] / d_Sh[i_R];

        dNew = d_Sh[i] - c_Sh[i_L] * coef_L * k1 - a_Sh[i_R] * coef_R * k2;
        yNew = y_Sh[i] - y_Sh[i_L] * coef_L * k1 - y_Sh[i_R] * coef_R * k2;
        aNew = -a_Sh[i_L]*  coef_L * k1;
        cNew = -c_Sh[i_R] * coef_R * k2;

        // Determine if this line has reached the bi-set
        __syncthreads();

        // Update
        d_Sh[i] = dNew;
        y_Sh[i] = yNew;
        a_Sh[i] = aNew;
        c_Sh[i] = cNew;
        delta *= 2;
        __syncthreads();
    }
    // Update of x_Sh

    x_Sh[idx] = y_Sh[idx] / d_Sh[idx];
	
    __syncthreads();
    //copy result from shared memory to x
    x[idx] = x_Sh[idx];
}



// *** CPU functions ***
int checker_CPU(const float *a, const float *d, const float *c, const float *y, float *x_found, unsigned int N)
{
	float eps = 10e-3f;

	if (abs(x_found[0] * d[0] + x_found[1] * c[0] - y[0]) > eps ||
		abs(x_found[N - 2] * a[N - 1] + x_found[N - 1] * d[N - 1] - y[N - 1]) > eps)
	{
		printf("\nProblem in: 0 or N-1\n");
		return 0;
	}
	for (int i = 1; i < N - 1; i++)
	{
		if (abs((x_found[i - 1] * a[i] + x_found[i] * d[i] + x_found[i + 1] * c[i]) - y[i]) > eps)
		{
			printf("\nProblem in: %d\n", i);
			return 0;
		}
	}
	return 1;
}

void thomas_CPU(const float *a, const float *d, const float *c, const float *y, float *x, unsigned int N)
{
	/* Allocate scratch space. */
	float* c_temp = NULL;
	c_temp = (float*)malloc(sizeof(float) * N);

	if (!c_temp)
	{
		printf("Error in allocation c_temp\n ");
		return;
	}

	c_temp[0] = c[0] / d[0];
	x[0] = y[0] / d[0];

	/* loop from 1 to N - 1 inclusive */
	for (int i = 1; i < N; i++)
	{
		const float dino = 1.0f / (d[i] - a[i] * c_temp[i - 1]);
		c_temp[i] = c[i] * dino;
		x[i] = (y[i] - a[i] * x[i - 1]) * dino;
	}

	/* loop from X - 2 to 0 inclusive, safely testing loop end condition */
	for (int i = N - 2; i > 0; i--)
		x[i] -= c_temp[i] * x[i + 1];
	x[0] -= c_temp[0] * x[1];

	/* free scratch space */
	free(c_temp);
}

int main(int argc, char *argv[])
{
	int N = atoi(argv[1]);
	// int N = 1023; // caution : don't work for CR because it's not a power of 2
	// int N = 3; // caution : don't work for CR because it's not a power of 2
	// int N = 16;
	float *a, *d, *c, *y, *x, *x_gpu_result, *x_result_CR, *x_result_PCR;
	float *a_gpu, *d_gpu, *c_gpu, *y_gpu, *x_gpu, *x_gpu2;
	//Timer timer;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//timer.start();

	//creat dump data to test
	a = (float *)malloc(N * sizeof(float));
	d = (float *)malloc(N * sizeof(float));
	c = (float *)malloc(N * sizeof(float));
	y = (float *)malloc(N * sizeof(float));
	x = (float *)malloc(N * sizeof(float));
	x_gpu_result = (float *)malloc(N * sizeof(float));
	x_result_CR = (float *)malloc(N * sizeof(float));
	x_result_PCR = (float *)malloc(N * sizeof(float));

	// Test to check if every methods give the same result
    for (int i = 0; i < N; i++)
    {
        a[i] = 1.2 * i + 2;
        d[i] = 5.1 * i + 5;
        c[i] = 2.2 * i + 2;
        y[i] = 5.0 + 2;
    } 

    // Test on a simple matrix 3x3 that can be solve with the LU method for example
    // with LU methode we find x = [49, -11.5, 4]
    /*
    a[1] = 2; a[2] = 6;
    d[0] = 1; d[1] = 10; d[2] = 18;
    c[0] = 4; c[1] = 5;
    y[0] = y[1] = y[2] = 3; */

	// Application to a deflection of a beam
	/*
    float N2 = N;
    float h = 1 / (N2 + 1);
    float H = h*h;
    float div_H = 1 / H;
    for (int i = 0; i < N; i++)
    {
        a[i] = - 1 * div_H;
        d[i] =  (2 + H) * div_H;
        c[i] = - 1 * div_H;
        y[i] = 1;
    } */

	
	//test the functions and time theme
	//Timer timer;
	float elapsed_time;

	cudaEventRecord(start, 0);
	
	thomas_CPU(a, d, c, y, x, N); //call the function
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("THOMAS METHOD -CPU IMPLEMENTATION-\n");
	printf("Elapsed time on CPU: %f ms", elapsed_time);

	printf("\tChecking: %s\n", checker_CPU(a, d, c, y, x, N)?"Success":"Fail"); //print 1 if Ax=y (success); 0 if not

	printf("\tResult on CPU: ");
	for (int i = 0; i < N; i++)
	printf("%f ", x[i]);
	
	printf("\n");

  

	//**** GPU ****
	// ********* Thomas methode **************

	cudaMalloc(&a_gpu, N * sizeof(float));
	cudaMalloc(&d_gpu, N * sizeof(float));
	cudaMalloc(&c_gpu, N * sizeof(float));
	cudaMalloc(&y_gpu, N * sizeof(float));
	cudaMalloc(&x_gpu, N * sizeof(float));
	//copy data to GPU (GPU global memory)
	cudaMemcpy(a_gpu, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gpu, d, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_gpu, c, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, N * sizeof(float), cudaMemcpyHostToDevice);

	// timer.start();
	// START
	cudaEventRecord(start, 0); // start recording at this time
	// *** Initial the kernel with just one thread as thomas method is sequential
	thomas_GPU<<<1, 1, sizeof(float) * N>>>(a_gpu, d_gpu, c_gpu, y_gpu, x_gpu, N);

	//stop recording and print the results
	cudaEventRecord(stop, 0); // stop enregistré à cet instant
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop); // temps en millisecondes

	printf("THOMAS METHOD -GPU IMPLEMENTATION-\n");
	printf("Elapsed time on GPU: %f ms", elapsed_time);

	//recopy result to host
	cudaMemcpy(x_gpu_result, x_gpu, N * sizeof(float), cudaMemcpyDeviceToHost);
	printf("\tChecking: %s\n", checker_CPU(a, d, c, y, x_gpu_result, N)?"Success":"Fail"); //print 1 if Ax=y (success); 0 if not

	printf("\tResult on CPU: ");
	for (int i = 0; i < N; i++)
		printf("%f ", x_gpu_result[i]);
	printf("\n");

		// ************* CR *************

		cudaMalloc(&a_gpu, N * sizeof(float));
		cudaMalloc(&d_gpu, N * sizeof(float));
		cudaMalloc(&c_gpu, N * sizeof(float));
		cudaMalloc(&y_gpu, N * sizeof(float));
		cudaMalloc(&x_gpu2, N * sizeof(float));
		//copy data to GPU (GPU global memory)
		cudaMemcpy(a_gpu, a, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_gpu, d, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(c_gpu, c, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(y_gpu, y, N * sizeof(float), cudaMemcpyHostToDevice);
	
		// START
		cudaEventRecord(start, 0); // start recording at this time
		// *** Initial the kernel with just one thread as thomas method is sequential
		CR<<<1, N/2, 5*N*sizeof(float)>>>(a_gpu, d_gpu, c_gpu, y_gpu, x_gpu2, N);
	
		//stop recording and print the results
		cudaEventRecord(stop, 0); // stop enregistré à cet instant
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_time, start, stop); // temps en millisecondes
	
		printf("CR METHOD -GPU IMPLEMENTATION-\n");
		printf("Elapsed time on GPU: %f ms", elapsed_time);
	
		//recopy result to host
		cudaMemcpy(x_result_CR, x_gpu2, N * sizeof(float), cudaMemcpyDeviceToHost);
		printf("\tChecking: %s\n", checker_CPU(a, d, c, y, x_result_CR, N)?"Success":"Fail"); //print 1 if Ax=y (success); 0 if not
	
		printf("\tResult on GPU: ");
		for (int i = 0; i < N; i++)
			printf("%f ", x_result_CR[i]);
		printf("\n");

	

		cudaFree(a_gpu);
		cudaFree(d_gpu);
		cudaFree(c_gpu);
		cudaFree(y_gpu);
		cudaFree(x_gpu2);

		//  ************* *PCR *************
		//find nb_step
		unsigned int N_power_of_2 = 1, nb_steps=0;

		while(true){
			N_power_of_2 *= 2;
			nb_steps ++;
			if(N_power_of_2 >= N)
				break;
		}
		printf("NB of step: %d *** \n", nb_steps);
		cudaMalloc(&a_gpu, N * sizeof(float));
		cudaMalloc(&d_gpu, N * sizeof(float));
		cudaMalloc(&c_gpu, N * sizeof(float));
		cudaMalloc(&y_gpu, N * sizeof(float));
		cudaMalloc(&x_gpu2, N * sizeof(float));
		//copy data to GPU (GPU global memory)
		cudaMemcpy(a_gpu, a, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_gpu, d, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(c_gpu, c, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(y_gpu, y, N * sizeof(float), cudaMemcpyHostToDevice);
	
		// START
		cudaEventRecord(start, 0); // start recording at this time
		// *** Initial the kernel with just one thread as thomas method is sequential
		printf("nb_step %d ****** N_po %d\n\n\n",nb_steps, N_power_of_2 );
		PCR<<<1, N, 5 * N * sizeof(float)>>>(a_gpu, d_gpu, c_gpu, y_gpu, x_gpu2, N, nb_steps);
	
		//stop recording and print the results
		cudaEventRecord(stop, 0); // stop enregistré à cet instant
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_time, start, stop); // temps en millisecondes
	
		printf("PCR METHOD -GPU IMPLEMENTATION-\n");
		printf("Elapsed time on GPU: %f ms", elapsed_time);
	
		//recopy result to host
		cudaMemcpy(x_result_PCR, x_gpu2, N * sizeof(float), cudaMemcpyDeviceToHost);
		printf("\tChecking: %s\n", checker_CPU(a, d, c, y, x_result_PCR, N)?"Success":"Fail"); //print 1 if Ax=y (success); 0 if not
	
		printf("\tResult on GPU: ");
		for (int i = 0; i < N; i++)
			printf("%f ", x_result_PCR[i]);
		printf("\n");

	
		cudaFree(a_gpu);
		cudaFree(d_gpu);
		cudaFree(c_gpu);
		cudaFree(y_gpu);
		cudaFree(x_gpu2);

	//free memory
	
	free(a);
	free(d);
	free(c);
	free(y);
	free(x);
	free(x_gpu_result);
}
