#include <bits/stdc++.h>

//using namespace std;


void matrix_vector_mul_cpu(float const * A, float const *b, float *c, size_t const NY,size_t const NX){

	//A: NY NXs

	for(size_t i = 0; i< NY; ++i)
		c[i] = std::inner_product( A+i*NX, A+(i+1)*NX, b, 0);

}




#include <cuda_runtime.h>
#include <cublas_v2.h>

void matrix_vector_mul_gpu(float const * a,float const * b,float * c,size_t const ha,size_t const wa)
{
	size_t const wb=1;

	const float alf = 1.0f;
	const float bet = 0.0f;
	const float *alpha = &alf;
	const float *beta = &bet;

	cublasStatus_t stat;
	cublasHandle_t handle;

	//If you need to do more than one matrix multiplication in your code it
	//is advisable to move the create/destroy handle code to the main function
	//, and use the same handle for all multiplications.
	stat = cublasCreate(&handle);
	assert(stat == CUBLAS_STATUS_SUCCESS);

	//suprising cublas use column major order, just for Fortran
	//reversed order of A and B passed to cublasSgemm
	//a good explanation:
	//http://mccormickml.com/2015/08/29/matrix-multiplication-with-cublas-example/
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(wb), static_cast<int>(ha), static_cast<int>(wa), alpha, b, static_cast<int>(wb), a, static_cast<int>(wa), beta, c, static_cast<int>(wb));

	cublasDestroy(handle);
}


#include <lib/tunable.h>
#include <lib/vectorpu.h>
#include <lib/meterpu.h>

typedef void (*matrix_vector_mul)(float const * ,float const * ,float * ,size_t const ,size_t const);
#define MATRIX_VECTOR_MUL_NUM_VARIANTS 2
#define MATRIX_VECTOR_MUL_NUM_DIM 2

template <class MeasureType, class ...Tunable_Args>
struct matrix_vector_mul_tunable : public tunable<MeasureType, matrix_vector_mul, Tunable_Args...>{
	matrix_vector_mul_tunable():tunable<MeasureType, matrix_vector_mul, Tunable_Args...>(
			MATRIX_VECTOR_MUL_NUM_DIM, MATRIX_VECTOR_MUL_NUM_VARIANTS, {matrix_vector_mul_cpu, matrix_vector_mul_gpu}
			){}

	std::vector<std::tuple<size_t, size_t, MeasureType, Tunable_Args...> > run(
			std::vector<bool> const& variant_mask, size_t const repeat_size, size_t const HA, size_t const WA) const{
			
		
			size_t const WB = 1;

			using namespace meterpu;
			//encapsulate the difference of correct meter
			meter<CPU_Time> cpu_meter;
			meter<CUDA_Time> cuda_meter;
			CPU_Time::ResultType val;

			vectorpu::vector<float> A(WA*HA,1), B(WA*WB,1), C(HA*WB,0), C_ref(HA*WB,WA);

			size_t const num_variant_run=count(variant_mask.cbegin(),variant_mask.cend(),true);
			std::vector<std::tuple<size_t, size_t, MeasureType, Tunable_Args...> > results;
			results.reserve(num_variant_run*repeat_size);

			//support by high performance sampling by reuse
			//the problem instance allocated
			for(size_t i=0; i<this->num_variants;
					++i){
				if(variant_mask[i]){
					for(size_t r=0; r<repeat_size; ++r){
						if(i<this->num_variants-1){
							cpu_meter.start();
							//encapsulate the difference of correct invoke
							(*this->dispatch_table[i])(R(A), R(B), W(C), HA, WA);
							cpu_meter.stop();
							cpu_meter.calc();
							val=cpu_meter.get_value();
						}
						else{
							cuda_meter.start();
							(*this->dispatch_table[i])(GR(A), GR(B), GW(C), HA, WA);
							cuda_meter.stop();
							cuda_meter.calc();
							val=cuda_meter.get_value();
						}
						results.emplace_back(r,i,val,HA,WA);
					}
				}
			}


			//encapsulate the difference of correctness check
			assert( equal(RI(C),REI(C), RI(C_ref)) );


			return results;

	}
};
