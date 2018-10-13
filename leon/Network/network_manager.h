#ifndef NETWORK_MANAGER_H
#define NETWORK_MANAGER_H
#include "network.h"
#include "network_defines.h"

class Network_Manager{
public:
	inline Network_Manager(){
		network_map = create_network();
		#ifdef PROFILE
		// for (int i = 0; i < 12; i++){
		// 	cpu_cycles[i] = new u64[network_map.size() - 1];
		// 	power_consumption[i] = new double[network_map.size() - 1];
		// }
		#elif defined(DUAL_CPU)
		//wake up leon RT
		#endif
	}

	~Network_Manager(){
		for (std::vector<Layers *>::iterator iter = network_map.begin(); iter != network_map.end(); iter++){
			delete (*iter);
		}
		network_map.clear();
		// #ifdef PROFILE
		// for (int i = 0; i < 12; i++){
		// 	delete [] cpu_cycles[i];
		// 	delete [] power_consumption[i];
		// }
		// #endif
	}

	void execute();
	void network_output();

	#ifdef PROFILE
	void shave_profile();
	void profile_output();
	#endif

private:
	std::vector<Layers *> network_map;
	u64 network_cycles = 0;
	// #ifdef PROFILE
	// 	double *power_consumption[12];
	// 	u64 *cpu_cycles[12];
	// #endif
};
#endif