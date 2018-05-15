
#include <iostream>
#include <assert.h>
#include <lib/tuneit.h>
#include <lib/matrix_vector_mul_tunable.h>

using namespace std;

int main()
{

	assert( cudaDeviceReset() == cudaSuccess ) ;



	size_t const depth=2;
	vector<bool> const mask(2,true);

	tuneit::tuneit_settings<MATRIX_VECTOR_MUL_NUM_DIM> st{depth, mask, true, false, true, 40, { {1,200}, {1,200} } };

	constexpr size_t num_vertices=4;

	tuneit::tuneit< MATRIX_VECTOR_MUL_NUM_VARIANTS, num_vertices, matrix_vector_mul_tunable<float, size_t, size_t>,
			float, size_t, size_t> mytuner(st);

	mytuner.train();

	cout<<"prediction is: "<<mytuner.predict(20,20)<<endl;;



	return EXIT_SUCCESS;
}
