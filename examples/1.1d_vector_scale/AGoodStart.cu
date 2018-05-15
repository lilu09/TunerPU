
#include <iostream>
#include <assert.h>
#include <lib/tuneit.h>
#include <lib/vector_scale_tunable.h>

using namespace std;

int main()
{

	assert( cudaDeviceReset() == cudaSuccess ) ;



	size_t const depth=2;
	vector<bool> const mask(2,true);

	tuneit::tuneit_settings<VECTOR_SCALE_NUM_DIM> st{depth, mask, true, false, true, 40, { {1,200} } };

	constexpr size_t num_vertices=2;

	tuneit::tuneit< VECTOR_SCALE_NUM_VARIANTS, num_vertices, vector_scale_tunable<float, size_t>,
			float, size_t> mytuner(st);

	mytuner.train();

	cout<<"prediction is: "<<mytuner.predict(20)<<endl;;



	return EXIT_SUCCESS;
}
