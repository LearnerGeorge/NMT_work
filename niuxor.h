#include "../../tensor/XGlobal.h"
#include "../../tensor/XTensor.h"
#include "../../tensor/core/CHeader.h"
using namespace nts;

namespace niuxor
{
	struct xorModel {
		XTensor weight1;
		XTensor weight2;
		XTensor b;

		int h_size;
		int devID;
	};
	struct xorNet {
		XTensor hidden_state1;
		XTensor hidden_state2;
		XTensor hidden_state3;

		XTensor output;
	};

	int niuxorMain(int argc, const char ** argv);
}

