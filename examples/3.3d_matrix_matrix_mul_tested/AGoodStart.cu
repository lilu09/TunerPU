
#include <iostream>
#include <assert.h>

#include <bits/stdc++.h>

/*using namespace std;*/

/*#include <lib/vectorpu.h>*/
#include <lib/matrix_mul_tunable.h>
#include <tuneit.h>
#include <meterpu.h>

#define OPT
#define BASE


//#include <cuda_runtime.h>
//#include <cublas_v2.h>

/*void matrix_mul_cublas(float const * const a,float const * const b,float * const c,size_t const ha,size_t const wa,size_t const wb)*/
/*{*/

/*const float alf = 1.0f;*/
/*const float bet = 0.0f;*/
/*const float *alpha = &alf;*/
/*const float *beta = &bet;*/

/*cublasStatus_t stat;*/
/*cublasHandle_t handle;*/

/*stat = cublasCreate(&handle);*/
/*assert(stat == CUBLAS_STATUS_SUCCESS);*/

/*cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(wb), static_cast<int>(ha), static_cast<int>(wa), alpha, b, static_cast<int>(wb), a, static_cast<int>(wa), beta, c, static_cast<int>(wb));*/

/*cublasDestroy(handle);*/
/*}*/

//8
int main(int argc, char* argv[])
{

	assert( cudaDeviceReset() == cudaSuccess );

	assert( argc == 2 );

	unsigned int const problem_size=std::atoi(argv[1]);


	/*const size_t ha=200, wa=200, wb=200;*/
	const size_t ha=problem_size, wa=problem_size, wb=problem_size;

/*#ifdef BASE*/

/*#endif*/

/*#ifdef OPT*/
	{



		tuneit::tuneit_settings<MATRIX_MUL_NUM_DIM, MATRIX_MUL_NUM_VARIANTS> st{2, std::vector<bool>(4,true), true, false, true, 5, {{1,10}, {1,10}, {1,10}} };

		tuneit::tuneit< MATRIX_MUL_NUM_VARIANTS, 8, matrix_mul_tunable<float, size_t, size_t, size_t>, float, size_t, size_t, size_t> mytuner(st);

		std::cout<<"before train"<<std::endl;

		mytuner.train();

		std::cout<<"after train"<<std::endl;

		vectorpu::vector<float> a(wa*ha,1), b(wa*wb,1), c(ha*wb,0);

		std::cout<<"before run"<<std::endl;

		mytuner.run(mytuner.predict(ha,wa,wb), a, b, c, ha, wa, wb);

		std::cout<<"after run"<<std::endl;

		using namespace meterpu;
		meterpu::meter<meterpu::CPU_Time> my_meter;

		for(size_t i=0;i<3;++i){
			std::cerr<<"Testing opt, problem size: "<<problem_size<<", No."<<i<<std::endl;
			my_meter.start();

			mytuner.run(mytuner.predict(ha,wa,wb), a, b, c, ha, wa, wb);

			my_meter.stop();
			my_meter.calc();
			std::cout<<my_meter.get_value()<<std::endl;
		}

		/*std::cout<<mytuner.predict(ha,wa,wb)<<std::endl;*/


		/*std::for_each(RI(c), REI(c), [](float const i){assert(i==200.0f);}) ;*/
		/*std::for_each(RI(c), REI(c), [](float const i){assert(i==10.0f);}) ;*/
		/*std::for_each(RI(c), REI(c), [problem_size](float const i){assert(i==float(problem_size));}) ;*/

	}
/*#endif*/




	return EXIT_SUCCESS;
}
