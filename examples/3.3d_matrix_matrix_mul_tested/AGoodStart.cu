
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


int main(int argc, char* argv[])
{

	assert( cudaDeviceReset() == cudaSuccess );

	assert( argc == 2 );

	unsigned int const problem_size=std::atoi(argv[1]);


	const size_t ha=problem_size, wa=problem_size, wb=problem_size;

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


	}




	return EXIT_SUCCESS;
}
