


void vector_scale_cpu(float *x, size_t N, float val){

	for(size_t i=0;i<N;++i)
		x[i] *= val;

}

__global__
void vector_scale_gpu_kernel(float *x, size_t N, float val){

	size_t i = threadIdx.x;
		x[i] *= val;

}

void vector_scale_gpu(float *x, size_t N, float val){

	vector_scale_gpu_kernel<<<1, N>>>(x, N, val);

}


#include <lib/tunable.h>
#include <lib/vectorpu.h>
#include <lib/meterpu.h>

typedef void (*vector_scale)(float *, size_t, float);
#define VECTOR_SCALE_NUM_VARIANTS 2
#define VECTOR_SCALE_NUM_DIM 1

template <class MeasureType, class ...Tunable_Args>
struct vector_scale_tunable : public tunable<MeasureType, vector_scale, Tunable_Args...>{
	vector_scale_tunable():tunable<MeasureType, vector_scale, Tunable_Args...>(
			VECTOR_SCALE_NUM_DIM, VECTOR_SCALE_NUM_VARIANTS, {vector_scale_cpu, vector_scale_gpu}
			){}

	std::vector<std::tuple<size_t, size_t, MeasureType, Tunable_Args...> > run(
			std::vector<bool> const& variant_mask, size_t const repeat_size, size_t const arg1 ) const{

			using namespace meterpu;
			//encapsulate the difference of correct meter
			meter<CPU_Time> cpu_meter;
			meter<CUDA_Time> cuda_meter;
			CPU_Time::ResultType val;

			vectorpu::vector<float> A(arg1,1);
			float scale_factor=2.0f;

			size_t const num_variant_run=count(variant_mask.cbegin(),variant_mask.cend(),true);
			std::vector<std::tuple<size_t, size_t, MeasureType, Tunable_Args...> > results;
			results.reserve(num_variant_run*repeat_size);

			for(size_t i=0; i<this->num_variants; ++i){
				if(variant_mask[i]){
					for(size_t r=0; r<repeat_size; ++r){
						if(i<this->num_variants-1){
							cpu_meter.start();
							//encapsulate the difference of correct invoke
							(*this->dispatch_table[i])(RW(A), arg1, scale_factor);
							cpu_meter.stop();
							cpu_meter.calc();
							val=cpu_meter.get_value();
						}
						else{
							cuda_meter.start();
							(*this->dispatch_table[i])(GRW(A),  arg1, scale_factor);
							cuda_meter.stop();
							cuda_meter.calc();
							val=cuda_meter.get_value();
						}
						results.emplace_back(r,i,val,arg1);
					}
				}
			}

			//assert( equal(RI(A),REI(A), RI(A_ref)) );

			return results;

	}
};
