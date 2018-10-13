#include <ddr_functions_exports.h>

FUNCPTR_T jumpTable(int i)
{
	struct lib_function func = lib[i];

	switch (func.category) {

	// Utility Functions
	case func_common:
		return func.cat.cm.func;
	case func_conv:
		return func.cat.conv.func;
	case func_pool:
		return func.cat.pool.func;
	case func_acc:
		return func.cat.acc.func;
	case func_fc:
		return func.cat.fc.func;
	case func_lrn:
		return func.cat.lrn.func;
	case func_im2col:
		return func.cat.im2col.func;
	}

	return 0;
}


