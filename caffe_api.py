#!/usr/bin/python2
import os
import sys
import subprocess
import signal
import time
import argparse
import caffe
import cv2
import FpConvert
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
import pprint
from itertools import product
import matplotlib.pyplot as plt
import copy
import random

def explore_parallelism(network_profile, prototxt, caffemodel, network_name):

	print '\033[1m' + '[API] Checking {0} for parallelism optimization...'.format(network_name) + '\033[0m'
	caffe.set_mode_cpu()
	network = caffe.Net(prototxt, caffemodel, caffe.TEST)
	parsible_net = caffe_pb2.NetParameter()
	try:
		text_format.Merge(open(prototxt).read(), parsible_net)
	except IOError:
		print '\033[91m' + '[API] Could not open prototxt file. Exiting...' + '\033[0m'
		sys.exit(2)

	id = 0
	split_id = 0
	dropout_id = 0
	relu_id = 0
	split_position = 0 
	branch_counter = 0

	linear_network = True

	top_to_id = []
	layer_name_list = []
	bottom_vector = []
	branch_list = []
	parallel_list = []

	for one_layer in parsible_net.layer: 

		bottom_vector = []
		if ''.join(map(str, one_layer.type)) != 'ReLU' and ''.join(map(str, one_layer.type)) != 'Dropout':
			top_to_id.append(''.join(map(str, one_layer.top)))
			layer_name_list.append(one_layer.name.replace('/', '_'))

			if ''.join(map(str, one_layer.type)) != 'Input':
				for previous in one_layer.bottom:
					for i, j in enumerate(top_to_id):
	  					if j == previous:
							bottom = i
					bottom_vector.append(bottom)

		if network.layers[id + split_id + dropout_id + relu_id].type == 'Split':
			split_id += 1
			split_position = id - 1
			linear_network = False
			
		if split_position > 0 and network.layers[id + split_id + dropout_id + relu_id].type != 'ReLU' and network.layers[id + split_id + dropout_id + relu_id].type != 'Dropout':
			if network.layers[id + split_id + dropout_id + relu_id].type != 'Concat':
				if bottom_vector[0] == split_position: 
					branch_counter += 1
					branch_list.append({'Layers': [id], 'Branch': branch_counter })
				else:
					branch_list[branch_counter - 1]['Layers'].append(id)
			else:
				branch_counter = 0
				split_position = 0
				parallel_list.append({'Branches': copy.deepcopy(branch_list), 'Best linear time': 0, 'Best linear energy': 0})
				del branch_list[:]

		if ''.join(map(str, one_layer.type)) == 'ReLU':
			relu_id += 1
		elif ''.join(map(str, one_layer.type)) == 'Dropout':
			dropout_id += 1
		else:
			id += 1

	del bottom_vector[:]
	del top_to_id[:]

	if linear_network:
		print '\033[1m' + '[API] {0} is Linear'.format(network_name) + '\033[0m'
		return 0
	try:
		network_profile_file = open(network_profile, "r")
	except IOError:
		print '\033[91m' + '[API] Could not open network profile csv. Exiting...' + '\033[0m'
		sys.exit(2)

	profile_list = []
	for line in network_profile_file:
		profile_list.append(line)
	del profile_list[0:2]

	network_profile_file.close()

	layer_list = []
	for item in profile_list:
		item = item.split('\t')
		layer_list.append({'Id': int(item[0]), 'Type': str(item[1]), 'Time': map(float, item[2:14]), 'Energy': [a * b for a,b in zip(map(float, item[15:27]), map(float, item[2:14]))]})

	del profile_list[:]

	layer_config_list = []
	csv_offset = 0
	best_block_time = 0
	best_block_energy = 0
	concat_id = 0

	for block in parallel_list:
		for branch in block['Branches']: 	
			for node in branch['Layers']: 	
				layer_config_list.append({'Layer Id': node - concat_id, 'Execution Time': layer_list[node - 1 + csv_offset - concat_id]['Time'], 'Time types': [layer_list[node - 1 + csv_offset - concat_id]['Type'] for i in range(12)], 'Energy': layer_list[node - 1 + csv_offset - concat_id]['Energy'], 'Energy types': [layer_list[node - 1 + csv_offset - concat_id]['Type'] for i in range(12)]})

				while layer_list[node + csv_offset - concat_id]['Id'] == node - concat_id:

					iterator = 0
					for t, e in zip(layer_list[node + csv_offset - concat_id]['Time'], layer_list[node + csv_offset - concat_id]['Energy']):
						
						if t < layer_config_list[len(layer_config_list) - 1]['Execution Time'][iterator]:

							layer_config_list[len(layer_config_list) - 1]['Execution Time'][iterator] = t
							layer_config_list[len(layer_config_list) - 1]['Time types'][iterator] = layer_list[node + csv_offset - concat_id]['Type']

						if e < layer_config_list[len(layer_config_list) - 1]['Energy'][iterator]:

							layer_config_list[len(layer_config_list) - 1]['Energy'][iterator] = e
							layer_config_list[len(layer_config_list) - 1]['Energy types'][iterator] = layer_list[node + csv_offset - concat_id]['Type']
						iterator += 1
					csv_offset += 1
				best_block_time += min(layer_config_list[len(layer_config_list) - 1]['Execution Time'])
				best_block_energy += min(layer_config_list[len(layer_config_list) - 1]['Energy'])
			del branch['Layers'][:]
			branch['Layers'] = copy.deepcopy(layer_config_list)
			del layer_config_list[:]
		block['Best linear time'] = best_block_time
		block['Best linear energy'] = best_block_energy
		best_block_time = 0
		best_block_energy = 0
		concat_id += 1

	del layer_list[:]

	lrt_list = []
	los_list = []
	network_config = []

	node_selection_strategies = ['full bull', 'next bear', 'couple bear', 'pivot bull', 'pivot bear', 'couple bull', 'next bull', 'full bear']
	overlap_selection_strategies = ['coupling dominant', 'pivot dominant']

	for block in parallel_list:

		total_branches = block['Branches'][len(block['Branches']) - 1]['Branch']
		lrt_branches = total_branches / 2
		los_branches = total_branches - lrt_branches
		block_config = []

		if lrt_branches == 2:
			for overlap_strategy in overlap_selection_strategies:
				print '\033[1m' + '\n[API] Trying \'{0}\' overlap matching strategy...'.format(overlap_strategy) + '\033[0m'
				for node_strategy in node_selection_strategies:
					print '\033[1m' + '[API] Trying \'{0}\' node selection strategy...'.format(node_strategy) + '\033[0m'
					for branch1 in block['Branches']:
						for branch2 in block['Branches']:
							if branch1['Branch'] >= branch2['Branch']:
								continue
							elif branch2['Branch'] > 3:
								continue
							else:

								branch_config_dict = {}
								branch_config_list = []
								lrt_list = copy.deepcopy([branch1, branch2])

								los_list = copy.deepcopy(block['Branches'])
								los_list.remove(block['Branches'][branch1['Branch'] - 1])
								los_list.remove(block['Branches'][branch2['Branch'] - 1])
								lrt_nodes = 0

								for lrt_branch in lrt_list:
									lrt_nodes += len(lrt_branch['Layers'])

								los_nodes = 0
								for los_branch in los_list:
									los_nodes += len(los_branch['Layers'])

								layers_executed = 0
								pivot_candidates = []
								los_execution = []
								lrt_execution = []

								while layers_executed < (los_nodes + lrt_nodes):

									shave_temp_lrt_list = []
									shave_temp_los_list = []
									del pivot_candidates[:]

									for branch_lrt, branch_los in zip(lrt_list, los_list):

										try:
											pivot_candidates.append({'Node': branch_lrt['Layers'][0], 'CPU': 'lrt'})
										except IndexError:
											pass
										try:
											pivot_candidates.append({'Node': branch_los['Layers'][0], 'CPU': 'los'})
										except IndexError:
											continue

									pivot_layer_time = pivot_candidates[0]['Node']['Execution Time'][0]
									pivot_layer = pivot_candidates[0]

									for layer in pivot_candidates:

										if node_strategy == 'full bull' or node_strategy == 'next bear' or node_strategy == 'couple bear' or node_strategy == 'pivot bull':
											if layer['Node']['Execution Time'][0] > pivot_layer_time:
												pivot_layer = layer
										else:
											if layer['Node']['Execution Time'][0] < pivot_layer_time:
												pivot_layer = layer										

									coupling_candidates_list = []

									if pivot_layer['CPU'] == 'lrt':

										for branch in los_list:
											try:
												coupling_candidates_list.append({'Node': branch['Layers'][0], 'CPU': 'los'})
											except IndexError:
												continue
										try:
											coupling_layer_time = coupling_candidates_list[0]['Node']['Execution Time'][0]
											coupling_layer = coupling_candidates_list[0]
										except IndexError:

											try:
												coupling_layer_time = coupling_candidates_list[1]['Node']['Execution Time'][0]
												coupling_layer = coupling_candidates_list[1]
											except IndexError:

												best_layer_time = pivot_layer['Node']['Execution Time'][0]
												shave_total = 1
												for shave_number, execution_time in enumerate(pivot_layer['Node']['Execution Time']):
													if execution_time < best_layer_time:
														best_layer_time = execution_time
														shave_total = shave_number + 1
												best_config_for_round = {'Round Time': best_layer_time, 'Execution overlap': '0%', 'LRT node': [pivot_layer['Node']], 'LOS node': [-1], 'LRT shave': shave_total, 'LOS shave': 0}
												branch_config_list.append(copy.deepcopy(best_config_for_round))
												
												for branch in lrt_list:
													for layer in branch['Layers']:
														if layer['Layer Id'] == pivot_layer['Node']['Layer Id']:
															branch['Layers'].remove(layer)

												layers_executed += 1
												continue

										coupling_layer_time = coupling_candidates_list[0]['Node']['Execution Time'][0]
										coupling_layer = coupling_candidates_list[0]

										for layer in coupling_candidates_list:

											if node_strategy == 'full bull' or node_strategy == 'next bear' or node_strategy == 'pivot bear' or node_strategy == 'couple bull':
												if layer['Node']['Execution Time'][0] > coupling_layer_time:
													coupling_layer = layer
											else:
												if layer['Node']['Execution Time'][0] < coupling_layer_time:
													coupling_layer = layer
										for branch in los_list:
											for layer in branch['Layers']:
												if layer['Layer Id'] == coupling_layer['Node']['Layer Id']:
													branch['Layers'].remove(layer)
													break
										for branch in lrt_list:
											for layer in branch['Layers']:
												if layer['Layer Id'] == pivot_layer['Node']['Layer Id']:
													branch['Layers'].remove(layer)

										pivot_time = 0
										coupling_time = 0
										best_time_for_round = 0
										best_config_for_round = {}

										for pivot_shaves in range(1, 12):

											los_id_list = []
											los_id_list.append(coupling_layer['Node'])
											shave_temp_lrt_list = copy.deepcopy(lrt_list)
											shave_temp_los_list = copy.deepcopy(los_list)

											pivot_time = pivot_layer['Node']['Execution Time'][pivot_shaves - 1]
											coupling_time = coupling_layer['Node']['Execution Time'][11 - pivot_shaves]
											smallest_time = 0

											next_inserted = False

											while coupling_time < pivot_time:

												next_inserted = True
												try:
													smallest_next_time = shave_temp_los_list[0]['Layers'][0]['Execution Time'][11 - pivot_shaves]
													smallest_next_layer = shave_temp_los_list[0]['Layers'][0]
													los_id_list.append(shave_temp_los_list[0]['Layers'][0])
												except IndexError:
													try:
														smallest_next_time = shave_temp_los_list[1]['Layers'][0]['Execution Time'][11 - pivot_shaves]
														smallest_next_layer = shave_temp_los_list[1]['Layers'][0]
														los_id_list.append(shave_temp_los_list[1]['Layers'][0]) 
													except IndexError:
														next_inserted = False
														break
												for branch in shave_temp_los_list:

													try:
														if node_strategy == 'next bear' or node_strategy == 'pivot bull' or node_strategy == 'couple bull' or node_strategy == 'full bear':
															if branch['Layers'][0]['Execution Time'][11 - pivot_shaves] < smallest_next_time:
																smallest_next_time = branch['Layers'][0]['Execution Time'][11 - pivot_shaves]
																smallest_next_layer = branch['Layers'][0]
																los_id_list[len(los_id_list) - 1] = branch['Layers'][0]
														else:
															if branch['Layers'][0]['Execution Time'][11 - pivot_shaves] > smallest_next_time:
																smallest_next_time = branch['Layers'][0]['Execution Time'][11 - pivot_shaves]
																smallest_next_layer = branch['Layers'][0]
																los_id_list[len(los_id_list) - 1] = branch['Layers'][0]
													except IndexError:
														continue

												for branch in shave_temp_los_list:
													for layer in branch['Layers']:
														if layer['Layer Id'] == smallest_next_layer['Layer Id']:
															branch['Layers'].remove(layer)

												coupling_time += smallest_next_time 
											#while end

											if overlap_strategy == 'pivot dominant' and next_inserted == True:
												coupling_time -= los_id_list[-1]['Execution Time'][11 - pivot_shaves]
												los_id_list = los_id_list[:-1]
											else:
												pass

											if pivot_shaves == 1:
												best_time_for_round = max(coupling_time, pivot_time)
												if coupling_time > pivot_time:
													execution_overlap = 100*pivot_time/coupling_time
												else:
													execution_overlap = 100*(coupling_time/pivot_time)
												best_config_for_round = {'Round Time': max(coupling_time, pivot_time), 'Execution overlap': '{0}%'.format(str(execution_overlap)), 'LRT node': [pivot_layer['Node']], 'LOS node': los_id_list, 'LRT shave': pivot_shaves, 'LOS shave': (12 - pivot_shaves)}
											else:

												if max(coupling_time, pivot_time) < best_time_for_round:
													best_time_for_round = max(coupling_time, pivot_time)
													if coupling_time > pivot_time:
														execution_overlap = 100*pivot_time/coupling_time
													else:
														execution_overlap = 100*(coupling_time/pivot_time)
													best_config_for_round = {'Round Time': max(coupling_time, pivot_time), 'Execution overlap': '{0}%'.format(str(execution_overlap)), 'LRT node': [pivot_layer['Node']], 'LOS node': los_id_list, 'LRT shave': pivot_shaves, 'LOS shave': (12 - pivot_shaves)}
										#for end
										branch_config_list.append(copy.deepcopy(best_config_for_round))
										layers_executed += len(best_config_for_round['LOS node']) + len(best_config_for_round['LRT node'])

										for executed_layer in best_config_for_round['LOS node']:
											for branch in los_list:
												for layer in branch['Layers']:
													if layer['Layer Id'] == executed_layer['Layer Id']:
														branch['Layers'].remove(layer)
														break
									else:

										for branch in lrt_list:
											try:
												coupling_candidates_list.append({'Node': branch['Layers'][0], 'CPU': 'lrt'})
											except IndexError:
												continue
										try:
											coupling_layer_time = coupling_candidates_list[0]['Node']['Execution Time'][0]
											coupling_layer = coupling_candidates_list[0]
										except IndexError:

											try:
												coupling_layer_time = coupling_candidates_list[1]['Node']['Execution Time'][0]
												coupling_layer = coupling_candidates_list[1]
											except IndexError:
												best_layer_time = pivot_layer['Node']['Execution Time'][0]
												shave_total = 1

												for shave_number, execution_time in enumerate(pivot_layer['Node']['Execution Time']):
													if execution_time < best_layer_time:
														best_layer_time = execution_time
														shave_total = shave_number + 1

												for branch in los_list:
													for layer in branch['Layers']:
														if layer['Layer Id'] == pivot_layer['Node']['Layer Id']:
															branch['Layers'].remove(layer)
												best_config_for_round = {'Round Time': best_layer_time,'LRT node': [-1], 'Execution overlap': '0%', 'LOS node': [pivot_layer['Node']], 'LRT shave': 0, 'LOS shave': shave_total}
												branch_config_list.append(copy.deepcopy(best_config_for_round))
												layers_executed += 1
												continue

										for layer in coupling_candidates_list:

											if node_strategy == 'full bull' or node_strategy == 'next bear' or node_strategy == 'pivot bear' or node_strategy == 'couple bull':
												if layer['Node']['Execution Time'][0] > coupling_layer_time:
													coupling_layer = layer
											else:
												if layer['Node']['Execution Time'][0] < coupling_layer_time:
													coupling_layer = layer
										for branch in lrt_list:
											for layer in branch['Layers']:
												if layer['Layer Id'] == coupling_layer['Node']['Layer Id']:
													branch['Layers'].remove(layer)
													break
										for branch in los_list:
											for layer in branch['Layers']:
												if layer['Layer Id'] == pivot_layer['Node']['Layer Id']:
													branch['Layers'].remove(layer)

										pivot_time = 0
										coupling_time = 0
										best_time_for_round = 0
										best_config_for_round = {}

										for pivot_shaves in range(1, 12):									
											
											lrt_id_list = []
											lrt_id_list.append(coupling_layer['Node'])
											shave_temp_lrt_list = copy.deepcopy(lrt_list)
											shave_temp_los_list = copy.deepcopy(los_list)
											pivot_time = pivot_layer['Node']['Execution Time'][pivot_shaves - 1]
											coupling_time = coupling_layer['Node']['Execution Time'][11 - pivot_shaves]
											smallest_time = 0

											next_inserted = False

											while coupling_time < pivot_time:

												next_inserted = True
												try:
													smallest_next_time = shave_temp_lrt_list[0]['Layers'][0]['Execution Time'][11 - pivot_shaves]
													smallest_next_layer = shave_temp_lrt_list[0]['Layers'][0]
													lrt_id_list.append(shave_temp_lrt_list[0]['Layers'][0])
												except IndexError:

													try:
														smallest_next_time = shave_temp_lrt_list[1]['Layers'][0]['Execution Time'][11 - pivot_shaves]
														smallest_next_layer = shave_temp_lrt_list[1]['Layers'][0]
														lrt_id_list.append(shave_temp_lrt_list[1]['Layers'][0])
													except IndexError:
														next_inserted = False
														break

												for branch in shave_temp_lrt_list:
													try:

														if node_strategy == 'next bear' or node_strategy == 'pivot bull' or node_strategy == 'couple bull' or node_strategy == 'full bear':
															if branch['Layers'][0]['Execution Time'][11 - pivot_shaves] < smallest_next_time:
																smallest_next_time = branch['Layers'][0]['Execution Time'][11 - pivot_shaves]
																smallest_next_layer = branch['Layers'][0]
																lrt_id_list[len(lrt_id_list) - 1] = branch['Layers'][0]
														else:

															if branch['Layers'][0]['Execution Time'][11 - pivot_shaves] < smallest_next_time:
																smallest_next_time = branch['Layers'][0]['Execution Time'][11 - pivot_shaves]
																smallest_next_layer = branch['Layers'][0]
																lrt_id_list[len(lrt_id_list) - 1] = branch['Layers'][0]
													except IndexError:
														continue

												for branch in shave_temp_lrt_list:
													for layer in branch['Layers']:
														if layer['Layer Id'] == smallest_next_layer['Layer Id']:
															branch['Layers'].remove(layer)

												coupling_time += smallest_next_time

											if overlap_strategy == 'pivot dominant' and next_inserted == True:

												coupling_time = coupling_time - lrt_id_list[-1]['Execution Time'][11 - pivot_shaves]
												lrt_id_list = lrt_id_list[:-1]

											else:
												pass

											if pivot_shaves == 1:

												best_time_for_round = max(coupling_time, pivot_time)
												if coupling_time > pivot_time:
													execution_overlap = 100*pivot_time/coupling_time
												else:
													execution_overlap = 100*(coupling_time/pivot_time)
												best_config_for_round = {'Round Time': max(coupling_time, pivot_time), 'Execution overlap': '{0}%'.format(str(execution_overlap)), 'LOS node': [pivot_layer['Node']], 'LRT node': lrt_id_list, 'LOS shave': pivot_shaves, 'LRT shave': (12 - pivot_shaves)}
											else:

												if max(coupling_time, pivot_time) < best_time_for_round:
													best_time_for_round = max(coupling_time, pivot_time)
													if coupling_time > pivot_time:
														execution_overlap = 100*pivot_time/coupling_time
													else:
														execution_overlap = 100*(coupling_time/pivot_time)
													best_config_for_round = {'Round Time': max(coupling_time, pivot_time), 'Execution overlap': '{0}%'.format(str(execution_overlap)), 'LOS node': [pivot_layer['Node']], 'LRT node': lrt_id_list, 'LOS shave': pivot_shaves, 'LRT shave': (12 - pivot_shaves)}

										branch_config_list.append(copy.deepcopy(best_config_for_round))
										layers_executed += len(best_config_for_round['LOS node']) + len(best_config_for_round['LRT node'])
										
										for executed_layer in best_config_for_round['LRT node']:
											for branch in lrt_list:
												for layer in branch['Layers']:
													if layer['Layer Id'] == executed_layer['Layer Id']:
														branch['Layers'].remove(layer)
														break

							branch_config_execution_time = 0

							for round in branch_config_list:
								branch_config_execution_time += round['Round Time']
							block_config.append(copy.deepcopy({'Branch config execution time': branch_config_execution_time, 'Round Configuration': branch_config_list, 'Node selection strategy': node_strategy, 'Overlap Matching strategy': overlap_strategy}))
		
		elif lrt_branches == 1:

			branch_config_dict = {}
			branch_config_list = []
			lrt_list = copy.deepcopy(block['Branches'][0])

			los_list = copy.deepcopy(block['Branches'][1])
			lrt_nodes = 0
			los_nodes = 0

			lrt_nodes += len(lrt_list['Layers'])
			los_nodes += len(los_list['Layers'])

			layers_executed = 0
			pivot_candidates = []
			los_execution = []
			lrt_execution = []

			while layers_executed < (los_nodes + lrt_nodes):

				shave_temp_lrt_list = []
				shave_temp_los_list = []
				del pivot_candidates[:]
				try:
					pivot_candidates.append({'Node': lrt_list['Layers'][0], 'CPU': 'lrt'})
				except IndexError:
					pass
				try:
					pivot_candidates.append({'Node': los_list['Layers'][0], 'CPU': 'los'})
				except IndexError:
					continue

				pivot_layer_time = pivot_candidates[0]['Node']['Execution Time'][0]
				pivot_layer = pivot_candidates[0]

				for layer in pivot_candidates:
					if layer['Node']['Execution Time'][0] > pivot_layer_time:
						pivot_layer = layer
				coupling_candidates_list = []

				if pivot_layer['CPU'] == 'lrt':
					try:
						coupling_candidates_list.append({'Node': los_list['Layers'][0], 'CPU': 'los'})

					except IndexError:
						continue
					try:
						coupling_layer_time = coupling_candidates_list[0]['Node']['Execution Time'][0]
						coupling_layer = coupling_candidates_list[0]

					except IndexError:
						best_layer_time = pivot_layer['Node']['Execution Time'][0]
						shave_total = 1

						for shave_number, execution_time in enumerate(pivot_layer['Node']['Execution Time']):
							if execution_time < best_layer_time:
								best_layer_time = execution_time
								shave_total = shave_number + 1

						best_config_for_round = {'Round Time': best_layer_time, 'Execution overlap': '0%', 'LRT node': [pivot_layer['Node']], 'LOS node': [-1], 'LRT shave': shave_total, 'LOS shave': 0}
						branch_config_list.append(copy.deepcopy(best_config_for_round))
						
						for layer in lrt_list['Layers']:
							if layer['Layer Id'] == pivot_layer['Node']['Layer Id']:
								lrt_list['Layers'].remove(layer)
								break

						layers_executed += 1
						continue

					coupling_layer_time = coupling_candidates_list[0]['Node']['Execution Time'][0]
					coupling_layer = coupling_candidates_list[0]

					for layer in coupling_candidates_list:
						if layer['Node']['Execution Time'][0] > coupling_layer_time:
							coupling_layer = layer

					for layer in los_list['Layers']:
						if layer['Layer Id'] == coupling_layer['Node']['Layer Id']:
							los_list['Layers'].remove(layer)
							break

					for layer in lrt_list['Layers']:
						if layer['Layer Id'] == pivot_layer['Node']['Layer Id']:
							lrt_list['Layers'].remove(layer)
							break
					pivot_time = 0
					coupling_time = 0
					best_time_for_round = 0
					best_config_for_round = {}

					for pivot_shaves in range(1, 12):

						los_id_list = []
						los_id_list.append(coupling_layer['Node'])
						shave_temp_lrt_list = copy.deepcopy(lrt_list)
						shave_temp_los_list = copy.deepcopy(los_list)

						pivot_time = pivot_layer['Node']['Execution Time'][pivot_shaves - 1]
						coupling_time = coupling_layer['Node']['Execution Time'][11 - pivot_shaves]
						smallest_time = 0
						while coupling_time < pivot_time:

							try:
								smallest_next_time = shave_temp_los_list[0]['Layers'][0]['Execution Time'][11 - pivot_shaves]
								smallest_next_layer = shave_temp_los_list[0]['Layers'][0]
								los_id_list.append(shave_temp_los_list[0]['Layers'][0])

							except (KeyError, IndexError) as exception:
								try:
									smallest_next_time = shave_temp_los_list[1]['Layers'][0]['Execution Time'][11 - pivot_shaves]
									smallest_next_layer = shave_temp_los_list[1]['Layers'][0]
									los_id_list.append(shave_temp_los_list[1]['Layers'][0])
								except (KeyError, IndexError) as exception:
									break
							try:
								if shave_temp_los_list['Layers'][0]['Execution Time'][11 - pivot_shaves] < smallest_next_time:
									smallest_next_time = shave_temp_los_list['Layers'][0]['Execution Time'][11 - pivot_shaves]
									smallest_next_layer = shave_temp_los_list['Layers'][0]
									los_id_list[len(los_id_list) - 1] = shave_temp_los_list['Layers'][0]
							except (KeyError, IndexError) as exception:
								continue

							for layer in shave_temp_los_list['Layers']:
								if layer['Layer Id'] == smallest_next_layer['Layer Id']:
									shave_temp_los_list['Layers'].remove(layer)
							coupling_time += smallest_next_time

						if pivot_shaves == 1:

							best_time_for_round = max(coupling_time, pivot_time)

							if coupling_time > pivot_time:
								execution_overlap = 100*pivot_time/coupling_time
							else:
								execution_overlap = 100*(coupling_time/pivot_time)

							best_config_for_round = {'Round Time': max(coupling_time, pivot_time), 'Execution overlap': '{0}%'.format(str(execution_overlap)), 'LRT node': [pivot_layer['Node']], 'LOS node': los_id_list, 'LRT shave': pivot_shaves, 'LOS shave': (12 - pivot_shaves)}
						else:

							if max(coupling_time, pivot_time) < best_time_for_round:

								best_time_for_round = max(coupling_time, pivot_time)
								if coupling_time > pivot_time:
									execution_overlap = 100*pivot_time/coupling_time
								else:
									execution_overlap = 100*(coupling_time/pivot_time)

								best_config_for_round = {'Round Time': max(coupling_time, pivot_time), 'Execution overlap': '{0}%'.format(str(execution_overlap)), 'LRT node': [pivot_layer['Node']], 'LOS node': los_id_list, 'LRT shave': pivot_shaves, 'LOS shave': (12 - pivot_shaves)}
					branch_config_list.append(copy.deepcopy(best_config_for_round))
					layers_executed += len(best_config_for_round['LOS node']) + len(best_config_for_round['LRT node'])

					for executed_layer in best_config_for_round['LOS node']:
						for layer in los_list['Layers']:
							if layer['Layer Id'] == executed_layer['Layer Id']:
								los_list['Layers'].remove(layer)
								break
				else:
					try:
						coupling_candidates_list.append({'Node': lrt_list['Layers'][0], 'CPU': 'lrt'})
					except IndexError:
						continue
					try:
						coupling_layer_time = coupling_candidates_list[0]['Node']['Execution Time'][0]
						coupling_layer = coupling_candidates_list[0]
					except IndexError:
						best_layer_time = pivot_layer['Node']['Execution Time'][0]
						shave_total = 1

						for shave_number, execution_time in enumerate(pivot_layer['Node']['Execution Time']):

							if execution_time < best_layer_time:
								best_layer_time = execution_time
								shave_total = shave_number + 1
						for layer in los_list['Layers']:

							if layer['Layer Id'] == pivot_layer['Node']['Layer Id']:
								los_list['Layers'].remove(layer)
						best_config_for_round = {'Round Time': best_layer_time, 'Execution overlap': '0%', 'LRT node': [-1], 'LOS node': [pivot_layer['Node']['Layer Id']], 'LRT shave': 0, 'LOS shave': shave_total}
						branch_config_list.append(copy.deepcopy(best_config_for_round))
						layers_executed += 1
						continue

					for layer in coupling_candidates_list:
						if layer['Node']['Execution Time'][0] > coupling_layer_time:
							coupling_layer = layer

					for layer in lrt_list['Layers']:
						if layer['Layer Id'] == coupling_layer['Node']['Layer Id']:
							lrt_list['Layers'].remove(layer)
							break

					for layer in los_list['Layers']:
						if layer['Layer Id'] == pivot_layer['Node']['Layer Id']:
							los_list['Layers'].remove(layer)
					pivot_time = 0
					coupling_time = 0
					best_time_for_round = 0
					best_config_for_round = {}

					for pivot_shaves in range(1, 12):

						lrt_id_list = []
						lrt_id_list.append(coupling_layer['Node'])
						shave_temp_lrt_list = copy.deepcopy(lrt_list)
						shave_temp_los_list = copy.deepcopy(los_list)
						pivot_time = pivot_layer['Node']['Execution Time'][pivot_shaves - 1]
						coupling_time = coupling_layer['Node']['Execution Time'][11 - pivot_shaves]
						smallest_time = 0

						while coupling_time < pivot_time:
							try:
								smallest_next_time = shave_temp_lrt_list['Layers'][0]['Execution Time'][11 - pivot_shaves]
								smallest_next_layer = shave_temp_lrt_list['Layers'][0]
								los_id_list.append(shave_temp_lrt_list['Layers'][0])
							except IndexError:
								break
							try:
								if shave_temp_lrt_list['Layers'][0]['Execution Time'][11 - pivot_shaves] < smallest_next_time:
									smallest_next_time = shave_temp_lrt_list['Layers'][0]['Execution Time'][11 - pivot_shaves]
									smallest_next_layer = shave_temp_lrt_list['Layers'][0]
									lrt_id_list[len(los_id_list) - 1] = shave_temp_lrt_list['Layers'][0]
							except IndexError:
								continue

							for layer in shave_temp_lrt_list['Layers']:
								if layer['Layer Id'] == smallest_next_layer['Layer Id']:
									shave_temp_lrt_list['Layers'].remove(layer)

							coupling_time += smallest_next_time

						if pivot_shaves == 1:
							best_time_for_round = max(coupling_time, pivot_time)
							if coupling_time > pivot_time:
								execution_overlap = 100*pivot_time/coupling_time
							else:
								execution_overlap = 100*(coupling_time/pivot_time)							
							best_config_for_round = {'Round Time': max(coupling_time, pivot_time), 'Execution overlap': '{0}%'.format(str(execution_overlap)), 'LOS node': [pivot_layer['Node']], 'LRT node': lrt_id_list, 'LOS shave': pivot_shaves, 'LRT shave': (12 - pivot_shaves)}
						else:
							if max(coupling_time, pivot_time) < best_time_for_round:
								best_time_for_round = max(coupling_time, pivot_time)
								if coupling_time > pivot_time:
									execution_overlap = 100*pivot_time/coupling_time
								else:
									execution_overlap = 100*(coupling_time/pivot_time)
								best_config_for_round = {'Round Time': max(coupling_time, pivot_time), 'Execution overlap': '{0}%'.format(str(execution_overlap)), 'LOS node': [pivot_layer['Node']], 'LRT node': lrt_id_list, 'LOS shave': pivot_shaves, 'LRT shave': (12 - pivot_shaves)}
					branch_config_list.append(copy.deepcopy(best_config_for_round))
					layers_executed += len(best_config_for_round['LOS node']) + len(best_config_for_round['LRT node'])
					for executed_layer in best_config_for_round['LRT node']:
						for layer in lrt_list['Layers']:
							if layer['Layer Id'] == executed_layer['Layer Id']:
								lrt_list['Layers'].remove(layer)
								break
			branch_config_execution_time = 0
			for round in branch_config_list:
				branch_config_execution_time += round['Round Time']
			block_config.append(copy.deepcopy({'Branch config execution time': branch_config_execution_time, 'Round Configuration': branch_config_list, 'Node selection strategy': 'None', 'Overlap Matching strategy': 'None'}))
		network_config.append(copy.deepcopy(block_config))


	optimal_parallel_time_list = []
	network_optimized = False

	for parallel_config, block in zip(network_config, parallel_list):

		optimal_time = block['Best linear time']
		optimal_parallel_time_list.append({})
		for branch_config in parallel_config:
			if branch_config['Branch config execution time'] < optimal_time:
				optimal_time = branch_config['Branch config execution time']
				optimal_parallel_time_list[len(optimal_parallel_time_list) - 1] = copy.deepcopy({'Best linear time': block['Best linear time'], 'configuration': branch_config})
				network_optimized = True

	if network_optimized == True:
		print '\033[92m' + '[API] Exploration on parallel nodes complete. Network optimized' + '\033[0m'
		return optimal_parallel_time_list
	else:
		print '\033[93m' + '[API] No parallel optimized configurations found!' + '\033[0m'
		return 0

def recursive_pareto_pruning(time_list, energy_list, operation_list, shave_list):

	pareto_points_length = 1

	for layer in time_list:
		pareto_points_length *= len(layer)

	time_config_list = []
	energy_config_list = []
	operation_config_list = []
	shave_config_list = []

	time_pareto_points = []
	energy_pareto_points = []
	operation_pareto_points = []
	shave_pareto_points = []

	if pareto_points_length < (12**4 + 1):

		for time_network_config in product(*time_list):
			time_sum = 0
			for time_layer_config in time_network_config:
				time_sum += time_layer_config
			time_config_list.append(time_sum)

		for energy_network_config in product(*energy_list):
			energy_sum = 0
			for energy_layer_config in energy_network_config:
				energy_sum += energy_layer_config
			energy_config_list.append(energy_sum)

		for operation_network_config in product(*operation_list):
			operation_sum = ''
			for operation_layer_config in operation_network_config:
				operation_sum += '{0}-'.format(str(operation_layer_config))
			operation_config_list.append(operation_sum)

		for shave_network_config in product(*shave_list):
			shave_sum = ''
			for shave_layer_config in shave_network_config:
				shave_sum += '{0}-'.format(str(shave_layer_config))
			shave_config_list.append(shave_sum)

		it = 0
		for operation, shave in zip(operation_config_list, shave_config_list):
			
			while '--' in operation:
				operation = operation.replace('--', '-')
			operation_config_list[it] = copy.deepcopy(operation)

			while '--' in shave:
				shave = shave.replace('--', '-')
			shave_config_list[it] = copy.deepcopy(shave)

			it += 1
		
		time, energy, operation, shave = (list(t) for t in zip(*sorted(zip(time_config_list, energy_config_list, operation_config_list, shave_config_list))))

		time_pareto_points.append(time[0])
		energy_pareto_points.append(energy[0])
		operation_pareto_points.append(operation[0])
		shave_pareto_points.append(shave[0])

		x_reference = energy[0]
		y_reference = time[0]

		for x, y, op, shv in zip(energy, time, operation, shave):
			if x < x_reference:

				energy_pareto_points.append(x)
				energy.remove(x)

				time_pareto_points.append(y)
				time.remove(y)

				operation_pareto_points.append(op)
				operation.remove(op)

				shave_pareto_points.append(shv)
				shave.remove(shv)

				x_reference = x
				y_reference = y
	else:

		recursive_time_list = []
		recursive_energy_list = []
		recursive_operation_list = []
		recursive_shave_list = []

		leaf_time_list = []
		leaf_energy_list = []
		leaf_operation_list = []
		leaf_shave_list = []

		leaf_time_list, leaf_energy_list, leaf_operation_list, leaf_shave_list = copy.deepcopy(recursive_pareto_pruning(time_list[0 : (len(time_list) / 2)], energy_list[0 : (len(energy_list) / 2)], operation_list[0 : (len(operation_list) / 2)], shave_list[0 : (len(shave_list) / 2)]))

		recursive_time_list.append(copy.deepcopy(leaf_time_list))
		del leaf_time_list[:]
		recursive_energy_list.append(copy.deepcopy(leaf_energy_list))
		del leaf_energy_list[:]
		recursive_operation_list.append(copy.deepcopy(leaf_operation_list))
		del leaf_operation_list[:]
		recursive_shave_list.append(copy.deepcopy(leaf_shave_list))
		del leaf_shave_list[:]

		leaf_time_list, leaf_energy_list, leaf_operation_list, leaf_shave_list = copy.deepcopy(recursive_pareto_pruning(time_list[(len(time_list) / 2) : len(time_list)], energy_list[(len(energy_list) / 2) : len(energy_list)], operation_list[(len(operation_list) / 2) : len(operation_list)], shave_list[(len(shave_list) / 2) : len(shave_list)]))
		
		recursive_time_list.append(copy.deepcopy(leaf_time_list))
		del leaf_time_list[:]
		recursive_energy_list.append(copy.deepcopy(leaf_energy_list))
		del leaf_energy_list[:]
		recursive_operation_list.append(copy.deepcopy(leaf_operation_list))
		del leaf_operation_list[:]
		recursive_shave_list.append(copy.deepcopy(leaf_shave_list))
		del leaf_shave_list[:]

		for time_network_config in product(*recursive_time_list):
			time_sum = 0
			for time_layer_config in time_network_config:
				time_sum += time_layer_config
			time_config_list.append(time_sum)

		for energy_network_config in product(*recursive_energy_list):
			energy_sum = 0
			for energy_layer_config in energy_network_config:
				energy_sum += energy_layer_config
			energy_config_list.append(energy_sum)

		for operation_network_config in product(*recursive_operation_list):
			operation_sum = ''
			for operation_layer_config in operation_network_config:
				operation_sum += '{0}-'.format(str(operation_layer_config))
			operation_config_list.append(operation_sum)

		for shave_network_config in product(*recursive_shave_list):
			shave_sum = ''
			for shave_layer_config in shave_network_config:
				shave_sum += '{0}-'.format(str(shave_layer_config))
			shave_config_list.append(shave_sum)

		it = 0
		for operation, shave in zip(operation_config_list, shave_config_list):
			
			while '--' in operation:
				operation = operation.replace('--', '-')
			operation_config_list[it] = copy.deepcopy(operation)

			while '--' in shave:
				shave = shave.replace('--', '-')
			shave_config_list[it] = copy.deepcopy(shave)

			it += 1
		
		time, energy, operation, shave = (list(t) for t in zip(*sorted(zip(time_config_list, energy_config_list, operation_config_list, shave_config_list))))

		time_pareto_points.append(time[0])
		energy_pareto_points.append(energy[0])
		operation_pareto_points.append(operation[0])
		shave_pareto_points.append(shave[0])

		x_reference = energy[0]
		y_reference = time[0]

		for x, y, op, shv in zip(energy, time, operation, shave):
			if x < x_reference:
				energy_pareto_points.append(x)
				time_pareto_points.append(y)
				operation_pareto_points.append(op)
				shave_pareto_points.append(shv)
				x_reference = x
				y_reference = y

	return time_pareto_points, energy_pareto_points, operation_pareto_points, shave_pareto_points

def layer_pareto_sub_optimals(layer_list):

	aligned_time_list = []
	aligned_energy_list = []
	time_config_list = []
	energy_config_list = []

	pareto_points_length = 1

	for layer in layer_list:

		layer_time_pareto_points = []
		layer_energy_pareto_points = []

		time, energy = (list(t) for t in zip(*sorted(zip(layer['Time'], layer['Energy']))))

		layer_time_pareto_points.append(time[0])
		layer_energy_pareto_points.append(energy[0])
		x_reference = energy[0]
		y_reference = time[0]

		for x, y in zip(energy, time):
			if x < x_reference:
				layer_energy_pareto_points.append(x)
				layer_time_pareto_points.append(y)
				x_reference = x
				y_reference = y

		del layer['Time'][:]
		del layer['Energy'][:]
		del layer['Operation'][:]
		del layer['shaves'][:]
		layer['Time'] = copy.deepcopy(layer_time_pareto_points)
		layer['Energy'] = copy.deepcopy(layer_energy_pareto_points)
		pareto_points_length *= len(layer['Time'])

	max_length = 1
	max_id = 0
	# for item in layer_list:
	# 	print item
	# 	print '\n\n\n'
	# print layer_list
	# print ''
	# print pareto_points_length

	while pareto_points_length > (12**6 + 1):

		max_length = 1
		max_id = -1
		candidate_id = 0

		while max_id == -1:
			candidate_id = random.randint(0, len(layer_list) - 1)
			if len(layer_list[candidate_id]['Time']) > 1:
				max_length = len(layer_list[candidate_id]['Time'])
				max_id = layer_list[candidate_id]['Id']

		# if max_length == 1:
		# 	break

		print 'diagrafo ton : ' + str(max_id)
		max_t_e_product = layer_list[candidate_id]['Time'][0]
		iterator = 0
		removed_point = 0

		for time, energy in zip(layer_list[candidate_id]['Time'], layer_list[candidate_id]['Energy']):
			if time * energy > max_t_e_product:
				max_t_e_product = time * energy
				removed_point = iterator
			iterator += 1

		removed_point = random.randint(0, max_length - 1)
		# print layer_list[max_id]
		print removed_point
		for layer in layer_list:
			if layer['Id'] == max_id:
				print layer['Time'][removed_point]
				print layer['Energy'][removed_point]
				del layer['Time'][removed_point]
				del layer['Energy'][removed_point]
				pareto_points_length = int(pareto_points_length / max_length) * (max_length - 1)

	for node in layer_list:
		aligned_time_list.append(node['Time'])
		aligned_energy_list.append(node['Energy'])

	for time_network_config in product(*aligned_time_list):
		time_sum = 0
		for time_layer_config in time_network_config:
			time_sum += time_layer_config
		time_config_list.append(time_sum)

	for energy_network_config in product(*aligned_energy_list):
		energy_sum = 0
		for energy_layer_config in energy_network_config:
			energy_sum += energy_layer_config
		energy_config_list.append(energy_sum)

	return time_config_list, energy_config_list

def network_exploration(network_profile, prototxt, caffemodel, network_name):

	parallel_config = explore_parallelism(network_profile, prototxt, caffemodel, network_name)
	parallel_config_list = []
	parallel_performance_gain = 0.0
	blocks_parallel_time = 0.0
	blocks_linear_time = 0.0

	if parallel_config != 0:
		parallel_config_list.append('High performance parallel configuration:\n\n')
		for block_iterator, block_opt_config in enumerate(parallel_config):

			if  bool(block_opt_config):
				parallel_config_list.append('CPU:\t\t\t\t\tLeon OS\t\t\t\t\t||\t\t\t\t\tLeon RT\n')
				parallel_config_list.append('Block {0}\n'.format(str(block_iterator + 1)))

				average_block_overlap = 0
				for round_iterator, round in enumerate(block_opt_config['configuration']['Round Configuration']):
					round_string = 'Round {0}:\t'.format(str(round_iterator + 1))

					for layer_iterator, los_layer in enumerate(round['LOS node']):
						if los_layer != -1:
							if layer_iterator < len(round['LOS node']) - 1:
								round_string += '[Id: {0}, Op: {1}, Shv: {2}] , '.format( str(los_layer['Layer Id']), str(los_layer['Time types'][round['LOS shave'] - 1]), str(round['LOS shave']))
							else:
								round_string += '[Id: {0}, Op: {1}, Shv: {2}]'.format( str(los_layer['Layer Id']), str(los_layer['Time types'][round['LOS shave'] - 1]), str(round['LOS shave']))						
						else:
							round_string += 'None'

					round_string += '\t||\t'

					for layer_iterator, lrt_layer in enumerate(round['LRT node']):
						if lrt_layer != -1:
							if layer_iterator < len(round['LRT node']) - 1:
								round_string += '[Id: {0}, Op: {1}, Shv: {2}] , '.format( str(lrt_layer['Layer Id']), str(lrt_layer['Time types'][round['LRT shave'] - 1]), str(round['LRT shave']))
							else:
								round_string += '[Id: {0}, Op: {1}, Shv: {2}]'.format( str(lrt_layer['Layer Id']), str(lrt_layer['Time types'][round['LRT shave'] - 1]), str(round['LRT shave']))						
						else:
							round_string += 'None'

					round_string += '\t-\t Execution overlap: {0}\n'.format(round['Execution overlap'])
					parallel_config_list.append(round_string)
					average_block_overlap += float(round['Execution overlap'].split('%')[0])

				average_block_overlap /= (round_iterator + 1)
				parallel_performance_gain += abs(block_opt_config['Best linear time'] - block_opt_config['configuration']['Branch config execution time'])
				parallel_config_list.append('\nBest Node overlap matching strategy for block: \'{0}\''.format(str(block_opt_config['configuration']['Overlap Matching strategy'])))
				parallel_config_list.append('\nBest Graph Traversal strategy for block: \'{0}\''.format(str(block_opt_config['configuration']['Node selection strategy'])))
				parallel_config_list.append('\nAverage Round Execution Overlap: {0}%'.format(str(average_block_overlap)))
				parallel_config_list.append('\n\nParallel execution time: {0} ms'.format(str(block_opt_config['configuration']['Branch config execution time'])))
				parallel_config_list.append('\nBest linear execution time: {0} ms'.format(str(block_opt_config['Best linear time'])))
				parallel_config_list.append('\nPerformance gain: {0}%'.format(str(100 * abs((block_opt_config['Best linear time'] - block_opt_config['configuration']['Branch config execution time']) / block_opt_config['Best linear time']))))
				parallel_config_list.append('\n---------------------------------------------------------\n')
				blocks_parallel_time += block_opt_config['configuration']['Branch config execution time']
				blocks_linear_time += block_opt_config['Best linear time']

	parallel_config_list.append('\n')

	print '\033[1m' + '[API] Generating pareto optimal points and plots...' + '\033[0m'

	try:
		network_profile_file = open(network_profile, "r")

	except IOError:
		print '\033[91m' + '[API] Could not open network profile csv. Exiting...' + '\033[0m'
		sys.exit(2)

	profile_list = []

	for line in network_profile_file:
		profile_list.append(line)
	del profile_list[0:2]
	try:
		network_profile_file.close()
	except IOError:
		print '\033[91m' + '[API] Could not close network profile csv. Exiting...' + '\033[0m'

	layer_list = []

	for iteration, line in enumerate(profile_list):
		line = line.split('\t')
		list_of_plane_points = []
		shaves_list = []
		time_list = []
		energy_list = []
		operation_list = []

		try:
			if int(line[0]) == layer_list[len(layer_list) - 1]['Id']:

				for line_offset in range(2, 14):
					shaves_list.append(line_offset - 1)
					operation_list.append(str(line[1]))
					time_list.append(float(line[line_offset]))
					energy_list.append(float(line[line_offset + 13]) * float(line[line_offset]))

				layer_list[len(layer_list) - 1]['shaves'] += shaves_list
				layer_list[len(layer_list) - 1]['Operation'] += operation_list
				layer_list[len(layer_list) - 1]['Time'] += time_list
				layer_list[len(layer_list) - 1]['Energy'] += energy_list
			else:

				for line_offset in range(2, 14):
					shaves_list.append(line_offset - 1)
					operation_list.append(str(line[1]))
					time_list.append(float(line[line_offset]))
					energy_list.append(float(line[line_offset + 13]) * float(line[line_offset]))
				layer_list.append({'Id': int(line[0]), 'shaves': shaves_list, 'Operation': operation_list, 'Time': time_list, 'Energy': energy_list})

		except IndexError:
			for line_offset in range(2, 14):
				shaves_list.append(line_offset - 1)
				operation_list.append(str(line[1]))
				time_list.append(float(line[line_offset]))
				energy_list.append(float(line[line_offset + 13]) * float(line[line_offset]))
			layer_list.append({'Id': int(line[0]), 'shaves': shaves_list, 'Operation': operation_list, 'Time': time_list, 'Energy': energy_list})

	del profile_list[:]

	pareto_points_length = 1
	for node in layer_list:
		pareto_points_length *= len(node['Time'])
	previous_length = 0

	pruned_time_points = []
	pruned_energy_points = []
	pruned_operation_points = []
	pruned_shave_points = []

	layer_time_sub_points = []
	layer_energy_sub_points = []
	layer_operation_sub_points = []
	layer_shave_sub_points = []

	aligned_time_list = []
	aligned_energy_list = []
	aligned_operation_list = []
	aligned_shave_list = []

	for node in layer_list:
		aligned_time_list.append(node['Time'])
		aligned_energy_list.append(node['Energy'])
		aligned_operation_list.append(node['Operation'])
		aligned_shave_list.append(node['shaves'])

	print '\033[1m' + '[API] Starting recursive pruning on design space...' + '\033[0m'

	pruned_time_points, pruned_energy_points, pruned_operation_points, pruned_shave_points = recursive_pareto_pruning(aligned_time_list, aligned_energy_list, aligned_operation_list, aligned_shave_list)

	# print '\033[1m' + '[API] Calculating layer sub optimal points for comparison...' + '\033[0m'

	# temp_time_list = []
	# temp_energy_list = []
	# temp_list = copy.deepcopy(layer_list)
	# for i in range(0, 12*12*12*2):
	# layer_time_sub_points, layer_energy_sub_points = layer_pareto_sub_optimals(layer_list)
	# 	layer_time_sub_points.extend(temp_time_list) 
	# 	layer_energy_sub_points.extend(temp_energy_list)
	# 	temp_list = copy.deepcopy(layer_list)

	# time_pareto_points = []
	# energy_pareto_points = []
	# operation_pareto_points = []
	# shave_pareto_points = []

	time_pareto_points, energy_pareto_points, operation_pareto_points, shave_pareto_points = (list(t) for t in zip(*sorted(zip(pruned_time_points, pruned_energy_points, pruned_operation_points, pruned_shave_points))))
	
	# energy_pareto_points.append(energy[0])
	# time_pareto_points.append(time[0])
	# operation_pareto_points.append(operation[0])
	# shave_pareto_points.append(shave[0])

	# x_reference = energy[0]
	# y_reference = time[0]

	# for x, y, op, shv in zip(energy, time, operation, shave):
	# 	if x < x_reference:
	# 		energy_pareto_points.append(x)
	# 		time_pareto_points.append(y)
	# 		operation_pareto_points.append(op)
	# 		shave_pareto_points.append(shv)
	# 		x_reference = x
	# 		y_reference = y

	if parallel_config != 0:
		parallel_config_list.append('Parallelism exploration summary:\n\nTotal parallel execution time: {0} ms\nBest total linear execution time: {1} ms\nBlock performance gain: {2}%\nTotal network performance gain: {3}%\n---------------------------------------------------------\n'.format(str(time_pareto_points[0] - parallel_performance_gain), str(time_pareto_points[0]), str(100*abs((blocks_linear_time - blocks_parallel_time) / blocks_linear_time)), str(100*(abs(parallel_performance_gain / time_pareto_points[0])))))
	
	print '\033[1m' + '[API] Appending data to exploration log file...' + '\033[0m'

	try:
		pareto_file = open('./{0}_pareto_config.log'.format(str(network_name)), 'w')
	except IOError:
		print '\033[91m' + '[API] Could not open descriptor for config log file. Exiting...' + '\033[0m'

	try:
		if parallel_config != 0:
			for line in parallel_config_list:
				pareto_file.write(line)
	except IOError:
		print '\033[91m' + '[API] Could not write optimal configurations log file. Exiting...' + '\033[0m'

	try:
		pareto_file.write('Pareto optimal linear shave configurations top-down from highest performance to highest efficiency\n\n')
		iterator = 1
		for pareto_time, pareto_energy, pareto_operation, pareto_shave in zip(time_pareto_points, energy_pareto_points, operation_pareto_points, shave_pareto_points):
			layer_id = 1
			if iterator == 1:
				pareto_file.write('High performance configuration:\n\n')
			elif iterator == len(time_pareto_points):
				pareto_file.write('High efficiency configuration:\n\n')

			for layer_operation_config, layer_shave_config in zip(pareto_operation.split('-')[0:len(pareto_operation.split('-')) - 1], pareto_shave.split('-')[0:len(pareto_shave.split('-')) - 1]):
				pareto_file.write('Layer {0}: {1} - {2} shaves\n'.format(str(layer_id), str(layer_operation_config), str(layer_shave_config)))
				layer_id += 1
			iterator += 1
			pareto_file.write('\nExecution Time: {0} ms\nEnergy Consumption: {1} mJ\n'.format(str(pareto_time), str(pareto_energy)))
			pareto_file.write('--------------------------------------------------\n\n')

	except IOError:
		print '\033[91m' + '[API] Could not write optimal configurations log file. Exiting...' + '\033[0m'

	try:
		pareto_file.close()
	except IOError:
		print '\033[91m' + '[API] Could not close descriptor for config log file. Exiting...' + '\033[0m'

	fig_pareto = plt.figure()
	plt.plot(layer_energy_sub_points, layer_time_sub_points, 'rx', mew = 0.5, ms = 4.0)
	plt.plot(energy_pareto_points, time_pareto_points, 'b', linewidth = '0.8')
	plt.plot(energy_pareto_points, time_pareto_points, 'bx', mew = 0.8, ms = 4.0, mfc = 'none')
	plt.grid(b = True, which = 'major', axis = 'both', linestyle = '--', linewidth = '0.6', animated = True)
	fig_pareto.suptitle('{0} Pareto Plot Time/Energy'.format(str(network_name)))
	plt.xlabel('Energy consumption (mJ)')
	plt.ylabel('Execution time (ms)')
	fig_pareto.savefig('./{0}_pareto_plot.png'.format(str(network_name)), dpi = 1200)
	
	print '\033[92m' + '[API] Profiling analysis complete!' + '\033[0m'

	high_performance_linear = {'Time': time_pareto_points[0], 'Energy': energy_pareto_points[0], 'Operation': operation_pareto_points[0].split('-')[0 : len(operation_pareto_points[0].split('-')) - 1], 'Shave': shave_pareto_points[0].split('-')[0 : len(shave_pareto_points[0].split('-')) - 1]}
	high_efficiency_linear = {'Time': time_pareto_points[-1], 'Energy': energy_pareto_points[-1], 'Operation': operation_pareto_points[-1].split('-')[0 : len(operation_pareto_points[0].split('-')) - 1], 'Shave': shave_pareto_points[-1].split('-')[0 : len(shave_pareto_points[0].split('-')) - 1]}

	return high_performance_linear, high_efficiency_linear, parallel_config


def compile_and_execute(clean_flag, server, network_name, profile_flag):

	files = ['weight_data.h', 'weight_data.c', 'network.cpp', 'network.h', 'network_defines.h']
	destination = ['./leon/Data/', './leon/Data/', './leon/Network/', './leon/Network/', './leon/Network/']
	for file, dest in zip(files, destination):
		try:
			process = subprocess.check_output('mv ./{0} {1}{2}'.format(file, dest, file).split())
		except subprocess.CalledProcessError:
			print '\033[91m' + '[API] Error on moving library files to folders. Exiting...' + '\033[0m'
			sys.exit(3)

	if server:
		try:
			fork_process = os.fork()
			if fork_process < 0:
				print '\033[91m' + "[API] Fork failed. Exiting..." + '\033[0m'
				sys.exit(4)

			server_PID = 0
			if fork_process == 0:
				try:

					print '\033[1m' + '\n[API] Forked child process. Initializing server...\n' + '\033[0m'
					process = subprocess.check_output('make start_server'.split())
					if server == 'keep_log':	
						print '\033[1m' + '[API] Writing server session to log file...' + '\033[0m'
						try:
							server_log_file = open('./server_output.log', 'w')
						except IOError:
							print '\033[91m' + '[API] Could not open server output file. Exiting...' + '\033[0m'
							sys.exit(2)
						server_log_file.write(process)
					time.sleep(0.5)
					print '\033[92m' + '[API] Child Process PID: {0} Execution Finished'.format(str(os.getpid())) + '\033[0m'
					sys.exit(0)
				except subprocess.CalledProcessError:
					print '\033[91m' + '[API] Server already initialized. Killing processes and exiting...' + '\033[0m'
					try:
						exec_pid = subprocess.check_output('pgrep moviDebugServer'.split())
						process = subprocess.Popen('kill {0}'.format(int(exec_pid)).split())
					except subprocess.CalledProcessError:
						pass
					try:
						make_pid = subprocess.check_output('pgrep make'.split())
						make_pid_list = make_pid.split('\n')
						for pid in make_pid_list:
							if pid != '':
								process = subprocess.Popen('kill {0}'.format(int(pid)).split())
					except subprocess.CalledProcessError:
						pass
					os.kill( os.getppid(), signal.SIGKILL)
					sys.exit(5)

			else:

				if clean_flag:
					process = subprocess.Popen('make clean'.split())
					process.wait()

					if args.profile != None:
						process = subprocess.Popen('make all PROFILE_MODE=yes'.split())
						process.wait()

						print '\033[1m' + '\n[API] Checking if server is up...\n' + '\033[0m'
						wait = True
						timeout = 0
						while wait:
							try:
								server_PID = subprocess.check_output('pgrep moviDebugServer'.split())
								wait = False
							except subprocess.CalledProcessError:
								time.sleep(0.6)
								timeout += 1
								if timeout > 100:
									print '\033[91m' + '[API] Server Timeout. Exiting...' + '\033[0m'
									os.kill(fork_process, signal.SIGTERM)
									sys.exit(6)

						process = subprocess.Popen('make run PROFILE_MODE=yes'.split())
						process.wait()
					
					else:	
						process = subprocess.Popen('make all'.split())
						process.wait()

						print '\033[1m' + '\n[API] Checking if server is up...\n' + '\033[0m'
						wait = True
						timeout = 0
						while wait:
							try:
								server_PID = subprocess.check_output('pgrep moviDebugServer'.split())
								wait = False
							except subprocess.CalledProcessError:
								time.sleep(0.6)
								timeout += 1
								if timeout > 100:
									print '\033[91m' + '[API] Server Timeout. Exiting...' + '\033[0m'
									os.kill(fork_process, signal.SIGTERM)
									sys.exit(6)

						process = subprocess.Popen('make run'.split())
						process.wait()
				else:
					if args.profile != None:
						process = subprocess.Popen('make all PROFILE_MODE=yes'.split())
						process.wait()

						print '\033[1m' + '\n[API] Checking if server is up...\n' + '\033[0m'
						wait = True
						timeout = 0
						while wait:
							try:
								server_PID = subprocess.check_output('pgrep moviDebugServer'.split())
								wait = False
							except subprocess.CalledProcessError:
								time.sleep(0.6)
								timeout += 1
								if timeout > 100:
									print '\033[91m' + '[API] Server Timeout. Exiting...' + '\033[0m'
									os.kill(fork_process, signal.SIGTERM)
									sys.exit(6)

						process = subprocess.Popen('make run PROFILE_MODE=yes'.split())
						process.wait()
					
					else:	
						process = subprocess.Popen('make all'.split())
						process.wait()

						print '\033[1m' + '\n[API] Checking if server is up...\n' + '\033[0m'
						wait = True
						timeout = 0
						while wait:
							try:
								server_PID = subprocess.check_output('pgrep moviDebugServer'.split())
								wait = False
							except subprocess.CalledProcessError:
								time.sleep(0.6)
								timeout += 1
								if timeout > 100:
									print '\033[91m' + '[API] Server Timeout. Exiting...' + '\033[0m'
									os.kill(fork_process, signal.SIGTERM)
									sys.exit(6)

						process = subprocess.Popen('make run'.split())
						process.wait()
			
				process = subprocess.Popen('kill {0}'.format(int(server_PID)).split())
				print '\033[92m' + '[API] Terminated successfully server with PID: {0}'.format(str(server_PID)) + '\033[0m'
				os.waitpid(fork_process, 0)
				print '\033[92m' + '[API] Parent Process PID: {0} Execution Finished'.format(str(os.getpid())) + '\033[0m'

		except KeyboardInterrupt:
			if server_PID != 0:
				process = subprocess.Popen('kill {0}'.format(int(server_PID)).split())
			if fork_process > 0:
				print '\033[91m' + '[API] Keyboard Interrupt. Exiting...' + '\033[0m'
				os.kill(fork_process, signal.SIGKILL)
			sys.exit(7)

	else:
		try:
			server_PID = subprocess.check_output('pgrep moviDebugServer'.split())
		except subprocess.CalledProcessError:
			print '\033[91m' + '[API] moviDebugServer is down. Exiting...' + '\033[0m'
			sys.exit(3)

		if clean_flag:
			process = subprocess.Popen('make clean'.split())
			process.wait()

			if args.profile != None:
				process = subprocess.Popen('make all PROFILE_MODE=yes'.split())
				process.wait()
				process = subprocess.Popen('make run PROFILE_MODE=yes'.split())
				process.wait()
					
			else:	
				process = subprocess.Popen('make all'.split())
				process.wait()

				process = subprocess.Popen('make run'.split())
				process.wait()
		else:
			if args.profile != None:
				process = subprocess.Popen('make all PROFILE_MODE=yes'.split())
				process.wait()
				process = subprocess.Popen('make run PROFILE_MODE=yes'.split())
				process.wait()
					
			else:	
				process = subprocess.Popen('make all'.split())
				process.wait()

				process = subprocess.Popen('make run'.split())
				process.wait()
	try:
		if args.profile != None:
			process = subprocess.check_output('mv ./network_profile.csv ./{0}_profile.csv'.format(network_name).split())
	except subprocess.CalledProcessError:
		print '\033[91m' + '[API] Error renaming csv network file. Exiting...' + '\033[0m'
		sys.exit(3)
	return

def library_generator(prototxt, caffemodel, image, layer_configuration, samples):

	#Assign to 'network' the full network list by caffe.Net
	caffe.set_mode_cpu()
	network = caffe.Net(prototxt, caffemodel, caffe.TEST)

	#parsible_net is a list containing the full network as well, but assigned by caffe_pb2.NetParameter method.
	#The difference between this method and caffe.Net, is that the latter apart from the standard layers contained
	#in the prototxt file, it also contains 'Split' Layer (see caffe docs) and also the blob size of each layer and
	#the weight values imported from caffemodel. The huge advantage though, of the first method, is that it 
	#contains directly the layer parameters specified by prototxt file. In the for loop below, both methods have
	#important role.

	parsible_net = caffe_pb2.NetParameter()
	try:
		text_format.Merge(open(prototxt).read(), parsible_net)
	except IOError:
		print '\033[91m' + 'Could not open prototxt file. Exiting...' + '\033[0m'
		sys.exit(2)

	###-------------------------------------------------------------------------------------------##

	###-------------------------------------------------------------------------------------------------------###
	#network.cpp contains the implementation of 'create_network' method
	#network.h is the header file to this method
	#weight_data.c contains the initialization of all weight arrays as well as the output buffers used by layers
	#weight_data.h externs all the buffers to make them available to methods

	network_list = []		#network_list will be written to network.cpp
	network_lib_list = []	#written to network.h
	network_defines_list = [] #network_defines.h
	weight_list = []		#written to weight_data.c
	weight_lib_list = []	#written to weight_data.h

	network_list.append("#include \"mv_types.h\"\n#include \"network.h\"\n#include \"weight_data.h\"\n#include \"ddr_functions_types.h\"\n\nstd::vector<Layers *> create_network(){\n\n    std::vector<Layers *> network_vector;\n")
	network_lib_list.append("#ifndef NETWORK_H\n#define NETWORK_H\n\n//------------------Includes---------------------//\n#include <vector>\n#include \"caffe_layers.h\"\n\n//------------------method declarations---------------------//\nstd::vector<Layers *> create_network();\n\n#endif")
		
	network_defines_list.append("#ifndef NETWORK_DEFINES_H\n#define NETWORK_DEFINES_H\n\n#ifdef PROFILE\n#define SAMPLES {0}\n#endif\n\n#define LINEAR TRUE\n\n#define INPUT FALSE\n\n#define CONVOLUTION FALSE\n\n#define POOLING FALSE\n\n#define INNERPRODUCT FALSE\n\n#define LRN FALSE\n\n#endif".format(str(samples)))

	weight_list.append("#include \"mv_types.h\"\n\n#define DDR_BUFFER __attribute__((section(\".ddr_direct.data\"), aligned (16)))\n\n")
	weight_lib_list.append("#ifndef WEIGHT_DATA_H\n#define WEIGHT_DATA_H\n\n")

	#weight_string contains the weight arrays
	#weight_library_string contains the extern commands of the weight arrays
	#output buffers will be appended in the end to weight_list
	weight_string = "//------------------weight data declaration---------------------//\n"
	weight_library_string = "//------------------weight data declaration---------------------//\n"
	###-------------------------------------------------------------------------------------------------------###


	###-------------------------------------------------------------------------------------------------------###
	#To be concurrent in each iteration and be certain that in both caffe.Net and caffe_pb2.NetParameter we are
	#referencing to the same layer, some Id integers are needed. id refers to all useful layers we will need to 
	#pass to our generated code. split_id exists in order to count the split layers and balance the abscence of
	#Split layers in caffe_pb2.NetParameter. dropout and relu id are used because we need to ignore these layers
	id = 0
	split_id = 0
	dropout_id = 0
	relu_id = 0
	concat_id = 0

	split_position = 0
	branch_counter = 0
	branch_buffer_index = 0
	network_list_offset = 1
	buffer_offset = 0

	top_to_id = []
	layer_name_list = []
	branch_list = []
	biggest_blob_list = [[0 for x in range(2)] for y in range(5)]

	refeed_flag = False
	if len(layer_configuration) > 1:
		refeed_flag = True

	#For loop runs onto .NetParameter() and not caffe.Net()
	#The reason is, we dont want to iterate over the extra layers added by the caffe Engine (split, dropout etc)
	#but only to those defined in the prototxt. Caffe.Net will be used to extract the weight arrays.
	for index, one_layer in enumerate(parsible_net.layer): 

		###-------------------------------------------------------------------------------------------------------###
		bottom_vector = []
		if ''.join(map(str, one_layer.type)) != 'ReLU' and ''.join(map(str, one_layer.type)) != 'Dropout':
			top_to_id.append(''.join(map(str, one_layer.top)))
			layer_name_list.append(one_layer.name.replace('/', '_'))
			#-----------------------------------------------#
			if ''.join(map(str, one_layer.type)) != 'Input':
				for previous in one_layer.bottom:
					for i, j in enumerate(top_to_id):
	  					if j == previous:
							bottom = i
					bottom_vector.append(bottom)
			#-----------------------------------------------#

		###-------------------------------------------------------------------------------------------------------###
		if network.layers[id + split_id + dropout_id + relu_id].type == 'Split': #if you find a split, then parallelism starts.
			split_id += 1	#You also need to take into consideration the split offset position
			split_position = id - 1 #split_position works as a boolean flag and as an integer indicating the position of the split layer at the same time
			split_buffer_index = branch_buffer_index
			
		if split_position > 0 and network.layers[id + split_id + dropout_id + relu_id].type != 'ReLU' and network.layers[id + split_id + dropout_id + relu_id].type != 'Dropout':

			if network.layers[id + split_id + dropout_id + relu_id].type != 'Concat': #while you don't find concat, parallelism keeps going on...

				if bottom_vector[0] == split_position: #If you are a child of split layer then just increment branch_counter and you have your buffer
					branch_buffer_index = 0
					branch_counter += 1
					branch_list.append({'Id': id, 'Branch': branch_counter, 'Index': branch_buffer_index})

				else: #if you are not a child of split layer
					for node in branch_list: #search your parent in the branch_list to get his branch_counter
						if node['Id'] == bottom_vector[0]:	#if this is true, then you found the dictionary of your father
							node['Id'] = id #you replace your father in the dictionary with yourself
							branch_buffer_index = node['Index'] ^ 1 #complement the index flag
							node['Index'] = branch_buffer_index #and replace it to your father's
							branch_counter = node['Branch'] #get your fathers branch_counter to write there

			else: #if concat then reset everything and return to linear computation of the network
				branch_counter = 0
				split_position = 0
				buffer_offset = 0
				del branch_list[:]

		###-------------------------------------------------------------------------------------------------------###



		###-------------------------------------------------------------------------------------------------------###

		################################################################		INPUT LAYER
		if ''.join(map(str, one_layer.type)) == 'Input':

			network_list.append("\n    network_vector.emplace_back(new Input(" 
								+ "(u8*)(&%s_input)" % one_layer.name.replace('/', '_').replace('-', '_') + ", "
								+ str(network.blobs[str(one_layer.name)].data.shape[1]) + ", " 
								+ str(network.blobs[str(one_layer.name)].data.shape[2]) + ", "
									+ str(network.blobs[str(one_layer.name)].data.shape[3]) + "));\n\n")

			if image != '':
				input_image = np.float32(cv2.normalize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY).astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX))
				list_array = np.empty((network.blobs[str(one_layer.name)].data.shape[1], network.blobs[str(one_layer.name)].data.shape[2], network.blobs[str(one_layer.name)].data.shape[3]))
				for i in range (0, network.blobs[str(one_layer.name)].data.shape[1]):
					list_array[i] = input_image

				weight_string += ("fp16 DDR_BUFFER " + one_layer.name.replace('/', '_').replace('-', '_') 
								+ "_input[{}]{};".format(str(network.blobs[str(one_layer.name)].data.shape[1]) + "*" 
								+ str(network.blobs[str(one_layer.name)].data.shape[2]) + "*"
								+ str(network.blobs[str(one_layer.name)].data.shape[3]), (" = {"  
								+ ', '.join(map(str, FpConvert.f32Tof16(list_array.ravel().tolist()))) + '}')) + "\n")
			else:
				weight_string += ("fp16 DDR_BUFFER " + one_layer.name.replace('/', '_').replace('-', '_') 
								+ "_input[{}]{};".format(str(network.blobs[str(one_layer.name)].data.shape[1]) + "*" 
								+ str(network.blobs[str(one_layer.name)].data.shape[2]) + "*"
								+ str(network.blobs[str(one_layer.name)].data.shape[3]), (" = {"  
								+ ', '.join(map(str, FpConvert.f32Tof16(list(network.blobs[one_layer.name].data[0].flatten())))) + '}')) + "\n")

			weight_library_string += ("extern u8 " + one_layer.name.replace('/', '_').replace('-', '_') + "_input[];\n")
			network_defines_list[0] = network_defines_list[0].replace("INPUT FALSE", "INPUT TRUE")
			id += 1

		################################################################		CONVOLUTION LAYER

		elif ''.join(map(str, one_layer.type)) == 'Convolution':

			local_counter = branch_counter
			local_buffer_index = branch_buffer_index
			buffer_offset_str = ''

			if split_position > 0:
				net_iterator = id + relu_id
				while ''.join(map(str, parsible_net.layer[net_iterator].type)) != 'Concat':
					net_iterator += 1
				for bottom in parsible_net.layer[net_iterator].bottom:
					if str(bottom) == one_layer.name:
						local_counter = 0
						local_buffer_index = split_buffer_index
						buffer_offset_str = '[' + str(2*buffer_offset) + ']'
						buffer_offset += network.blobs[str(one_layer.name)].data.shape[1]*network.blobs[str(one_layer.name)].data.shape[2]*network.blobs[str(one_layer.name)].data.shape[3]

			if network.blobs[str(one_layer.name)].data.shape[1]*network.blobs[str(one_layer.name)].data.shape[2]*network.blobs[str(one_layer.name)].data.shape[3] > biggest_blob_list[local_counter][local_buffer_index]:
				biggest_blob_list[local_counter][local_buffer_index] = network.blobs[str(one_layer.name)].data.shape[1]*network.blobs[str(one_layer.name)].data.shape[2]*network.blobs[str(one_layer.name)].data.shape[3]
				
			if ''.join(map(str, one_layer.convolution_param.stride)) == '':
				stride = "1"
			else:
				stride = ''.join(map(str, one_layer.convolution_param.stride))
			if ''.join(map(str, one_layer.convolution_param.pad)) == '':
				pad = "0"
			else:
				pad = ''.join(map(str, one_layer.convolution_param.pad))

			ddr_function = 0

			if ''.join(map(str, one_layer.convolution_param.kernel_size)) == '1':
				if stride == '1':
					ddr_function = 0
				else:
					print '\033[91m' + '[API] Error: Kernel 1 with stride {1} not supported!'.format(stride) + '\033[0m'
					sys.exit(6)
			elif ''.join(map(str, one_layer.convolution_param.kernel_size)) == '3':
				if int(stride) != 8:
					ddr_function = int(stride)
				elif int(stride) == 8:
					ddr_function = 5
				else:
					print '\033[91m' + '[API] Error: Kernel 3 with stride {1} not supported!'.format(stride) + '\033[0m'
					sys.exit(6)					
			elif ''.join(map(str, one_layer.convolution_param.kernel_size)) == '5':
				if int(stride) != 8:
					ddr_function = int(stride) + 5
				elif int(stride) == 8:
					ddr_function = 10
				else:
					print '\033[91m' + '[API] Error: Kernel 5 with stride {1} not supported!'.format(stride) + '\033[0m'
					sys.exit(6)	
			elif ''.join(map(str, one_layer.convolution_param.kernel_size)) == '7':
				if int(stride) == 1 or int(stride) == 2:
					ddr_function = int(stride) + 10
				elif int(stride) == 4:
					ddr_function = 13
				elif int(stride) == 8:
					ddr_function = 14 
				else:
					print '\033[91m' + '[API] Error: Kernel 7 with stride {1} not supported!'.format(stride) + '\033[0m'
					sys.exit(6)	
			elif ''.join(map(str, one_layer.convolution_param.kernel_size)) == '9':
				if int(stride) != 8:
					ddr_function = int(stride) + 14
				elif int(stride) == 8:
					ddr_function = 19
				else:
					print '\033[91m' + '[API] Error: Kernel 9 with stride {1} not supported!'.format(stride) + '\033[0m'
					sys.exit(6)					
			elif ''.join(map(str, one_layer.convolution_param.kernel_size)) == '11':
				if int(stride) == 1 or int(stride) == 2:
					ddr_function = int(stride) + 19
				elif int(stride) == 4:
					ddr_function = 22
				elif int(stride) == 8:
					ddr_function = 23 
				else:
					print '\033[91m' + '[API] Error: Kernel 11 with stride {1} not supported!'.format(stride) + '\033[0m'
					sys.exit(6)
			else:
				print '\033[91m' + '[API] Error: Convolution Kernel not supported!' + '\033[0m'
				sys.exit(6)

			if refeed_flag == True and layer_configuration['Operation'][id - concat_id - 1] != 'Conv_Direct':
				ddr_function = 46

			shaves_used = 0
			if refeed_flag == True:
				shaves_used = layer_configuration['Shave'][id - concat_id - 1]
			else:
				shaves_used = layer_configuration[0]

			network_list.append("    network_vector.emplace_back(new Convolution(" 
								+ "(u8*)(&branch_output_buffer_{0}_{1}{2})".format(str(local_counter), str(local_buffer_index), buffer_offset_str) + ", "
								+ str(bottom_vector[0]) + ", "
								+ str(shaves_used) + ", "
								+ "(u8*)(&%s_weights)" % one_layer.name.replace('/', '_').replace('-', '_') + ", "
								+ "(u8*)(&%s_biases)" % one_layer.name.replace('/', '_').replace('-', '_') + ", "
								+ str(network.blobs[str(one_layer.name)].data.shape[1]) + ", " 
								+ str(network.blobs[str(one_layer.name)].data.shape[2]) + ", " 
								+ str(network.blobs[str(one_layer.name)].data.shape[3]) + ", " 
								+ ''.join(map(str, one_layer.convolution_param.kernel_size)) + ", " 
								+ stride + ", " + pad + ", " + str(one_layer.convolution_param.group) + ", " + str(ddr_function) + ", 0));\n\n")
			
			weight_string += ("fp16 DDR_BUFFER " + one_layer.name.replace('/', '_').replace('-', '_')
								+ "_weights[{}]{};".format('*'.join(map(str, network.layers[id + split_id + dropout_id + relu_id].blobs[0].data.shape)), (" = {"  
								+ ', '.join(map(str, FpConvert.f32Tof16(list(network.layers[id + split_id + dropout_id + relu_id].blobs[0].data.flatten())))) 
								+ '}')) + "\n")
				
			weight_string += ("fp16 DDR_BUFFER " + one_layer.name.replace('/', '_').replace('-', '_')
								+ "_biases[{}]{};".format('*'.join(map(str, network.layers[id + split_id + dropout_id + relu_id].blobs[1].data.shape)), (" = {"  
								+ ', '.join(map(str, FpConvert.f32Tof16(list(network.layers[id + split_id + dropout_id + relu_id].blobs[1].data.flatten())))) 
								+ '}')) + "\n")
			weight_library_string += ("extern u8 " + one_layer.name.replace('/', '_').replace('-', '_') + "_weights[];\n")
			weight_library_string += ("extern u8 " + one_layer.name.replace('/', '_').replace('-', '_') + "_biases[];\n")

			branch_buffer_index = branch_buffer_index ^ 1
			network_defines_list[0] = network_defines_list[0].replace("CONVOLUTION FALSE", "CONVOLUTION TRUE")
			id += 1


		################################################################		POOLING LAYER

		elif ''.join(map(str, one_layer.type)) == 'Pooling':

			local_counter = branch_counter
			local_buffer_index = branch_buffer_index
			buffer_offset_str = ''

			if split_position > 0:
				net_iterator = id + relu_id
				while ''.join(map(str, parsible_net.layer[net_iterator].type)) != 'Concat':
					net_iterator += 1
				for bottom in parsible_net.layer[net_iterator].bottom:
					if str(bottom) == one_layer.name:
						local_counter = 0
						local_buffer_index = split_buffer_index
						buffer_offset_str = '[' + str(2*buffer_offset) + ']'
						buffer_offset += network.blobs[str(one_layer.name)].data.shape[1]*network.blobs[str(one_layer.name)].data.shape[2]*network.blobs[str(one_layer.name)].data.shape[3]
				
			if network.blobs[str(one_layer.name)].data.shape[1]*network.blobs[str(one_layer.name)].data.shape[2]*network.blobs[str(one_layer.name)].data.shape[3] > biggest_blob_list[local_counter][local_buffer_index]:
				biggest_blob_list[local_counter][local_buffer_index] = network.blobs[str(one_layer.name)].data.shape[1]*network.blobs[str(one_layer.name)].data.shape[2]*network.blobs[str(one_layer.name)].data.shape[3]


			if str(one_layer.pooling_param.stride) == '':
				stride = "1"
			else:
				stride = str(one_layer.pooling_param.stride)
			if str(one_layer.pooling_param.pad) == '':
				pad = "0"
			else:
				pad = str(one_layer.pooling_param.pad)
			if str(one_layer.pooling_param.pool) == '' or str(one_layer.pooling_param.pool) == '0':
				pool = "pooling_MAX"
			elif str(one_layer.pooling_param.pool) == '1':
				pool = "pooling_AVE"
			elif str(one_layer.pooling_param.pool) == '2':
				pool = "pooling_STOCHASTIC"
				print '\033[91m' + '[API] Error: Stochastic pooling not yet supported' + '\033[0m'
				sys.exit(6)
			else:
				print '\033[91m' + '[API] Error: Pooling Type Unknown' + '\033[0m'
				sys.exit(6)

			ddr_function = 0
			kernel_size = str(one_layer.pooling_param.kernel_size)

			if pool == 'pooling_AVE':
				if str(one_layer.pooling_param.kernel_size) == '3' and stride == '2':
					ddr_function = 28
				elif str(one_layer.pooling_param.kernel_size) == '7' and stride == '1':
					ddr_function = 29
				elif one_layer.pooling_param.global_pooling == True:
					ddr_function = 30
					kernel_size = '14'
				else:
					ddr_function = 29 #TODO now only for kernel 6 of imagenet
					print '\033[91m' + '[API] Error: Unsupported AVE pooling kernel size / stride' + '\033[0m'
					# sys.exit(6)
			elif pool == 'pooling_MAX':
				if str(one_layer.pooling_param.kernel_size) == '2' and stride == '2':
					ddr_function = 31
				elif str(one_layer.pooling_param.kernel_size) == '3' and stride == '1':
					ddr_function = 32
				elif str(one_layer.pooling_param.kernel_size) == '3' and stride == '2':
					if str(parsible_net.layer[index + 1].top) == str(one_layer.top) and str(parsible_net.layer[index + 1].type) == 'ReLU':
						ddr_function = 34
					else:
						ddr_function = 33
				else:
					print '\033[91m' + '[API] Error: Unsupported MAX kernel size / stride' + '\033[0m'
					sys.exit(6)
			else:
				print '\033[91m' + '[API] Error: Unsupported pooling type' + '\033[0m'
				sys.exit(6)

			shaves_used = 0
			if refeed_flag == True:
				shaves_used = layer_configuration['Shave'][id - concat_id - 1]
			else:
				shaves_used = layer_configuration[0]

			network_list.append("    network_vector.emplace_back(new Pooling(" 
								+ "(u8*)(&branch_output_buffer_{0}_{1}{2})".format(str(local_counter), str(local_buffer_index), buffer_offset_str) + ", "
								+ str(bottom_vector[0]) + ", "
								+ str(shaves_used) + ", "
								+ str(network.blobs[str(one_layer.name)].data.shape[1]) + ", " 
								+ str(network.blobs[str(one_layer.name)].data.shape[2]) + ", " 
								+ str(network.blobs[str(one_layer.name)].data.shape[3]) + ", " 
								+ kernel_size + ", "
								+ stride + ", " 
								+ pad + ", " 
								+ str(ddr_function) + ", "
								+ pool + "));\n\n")
			
			branch_buffer_index = branch_buffer_index ^ 1
			network_defines_list[0] = network_defines_list[0].replace("POOLING FALSE", "POOLING TRUE")
			id += 1

		################################################################		INNER PRODUCT LAYER

		elif ''.join(map(str, one_layer.type)) == 'InnerProduct':

			local_counter = branch_counter
			local_buffer_index = branch_buffer_index
			buffer_offset_str = ''

			if split_position > 0:
				net_iterator = id + relu_id
				while ''.join(map(str, parsible_net.layer[net_iterator].type)) != 'Concat':
					net_iterator += 1
				for bottom in parsible_net.layer[net_iterator].bottom:
					if str(bottom) == one_layer.name:
						local_counter = 0
						local_buffer_index = split_buffer_index
						buffer_offset_str = '[' + str(2*buffer_offset) + ']'
						buffer_offset += network.blobs[str(one_layer.name)].data.shape[0]*network.blobs[str(one_layer.name)].data.shape[1]

			if network.blobs[str(one_layer.name)].data.shape[0]*network.blobs[str(one_layer.name)].data.shape[1] > biggest_blob_list[branch_counter][branch_buffer_index]:
				biggest_blob_list[branch_counter][branch_buffer_index] = network.blobs[str(one_layer.name)].data.shape[0]*network.blobs[str(one_layer.name)].data.shape[1]

			shaves_used = 0
			if refeed_flag == True:
				shaves_used = layer_configuration['Shave'][id - concat_id - 1]
			else:
				shaves_used = layer_configuration[0]

			network_list.append("    network_vector.emplace_back(new InnerProduct(" 
								+ "(u8*)(&branch_output_buffer_{0}_{1}{2})".format(str(local_counter), str(local_buffer_index), buffer_offset_str) + ", "
								+ str(bottom_vector[0]) + ", "
								+ str(shaves_used) + ", "
								+ "(u8*)(&%s_weights)" % one_layer.name.replace('/', '_').replace('-', '_') + ", "
								+ "(u8*)(&%s_biases)" % one_layer.name.replace('/', '_').replace('-', '_') + ", "
								+ str(network.blobs[str(one_layer.name)].data.shape[1]) + ", " 
								+ "0));\n\n")
				
			weight_string += ("fp16 DDR_BUFFER " + one_layer.name.replace('/', '_').replace('-', '_') 
								+ "_weights[{}]{};".format('*'.join(map(str, network.layers[id + split_id + dropout_id + relu_id].blobs[0].data.shape)), (" = {"  
								+ ', '.join(map(str, FpConvert.f32Tof16(list(network.layers[id + split_id + dropout_id + relu_id].blobs[0].data.flatten())))) 
								+ '}')) + "\n")

			weight_string += ("fp16 DDR_BUFFER " + one_layer.name.replace('/', '_').replace('-', '_') 
								+ "_biases[{}]{};".format('*'.join(map(str, network.layers[id + split_id + dropout_id + relu_id].blobs[1].data.shape)), (" = {"  
								+ ', '.join(map(str, FpConvert.f32Tof16(list(network.layers[id + split_id + dropout_id + relu_id].blobs[1].data.flatten())))) 
								+ '}')) + "\n")
			weight_library_string += ("extern u8 " + one_layer.name.replace('/', '_').replace('-', '_') + "_weights[];\n")
			weight_library_string += ("extern u8 " + one_layer.name.replace('/', '_').replace('-', '_') + "_biases[];\n")

			branch_buffer_index = branch_buffer_index ^ 1
			network_defines_list[0] = network_defines_list[0].replace("INNERPRODUCT FALSE", "INNERPRODUCT TRUE")
			id += 1
			
		################################################################		LRN LAYER

		elif ''.join(map(str, one_layer.type)) == 'LRN':
				
			local_counter = branch_counter
			local_buffer_index = branch_buffer_index
			buffer_offset_str = ''

			if split_position > 0:
				net_iterator = id + relu_id
				while ''.join(map(str, parsible_net.layer[net_iterator].type)) != 'Concat':
					net_iterator += 1
				for bottom in parsible_net.layer[net_iterator].bottom:
					if str(bottom) == one_layer.name:
						local_counter = 0
						local_buffer_index = split_buffer_index
						buffer_offset_str = '[' + str(2*buffer_offset) + ']'
						buffer_offset += network.blobs[str(one_layer.name)].data.shape[1]*network.blobs[str(one_layer.name)].data.shape[2]*network.blobs[str(one_layer.name)].data.shape[3]
					
			if network.blobs[str(one_layer.name)].data.shape[1]*network.blobs[str(one_layer.name)].data.shape[2]*network.blobs[str(one_layer.name)].data.shape[3] > biggest_blob_list[local_counter][local_buffer_index]:
				biggest_blob_list[local_counter][local_buffer_index] = network.blobs[str(one_layer.name)].data.shape[1]*network.blobs[str(one_layer.name)].data.shape[2]*network.blobs[str(one_layer.name)].data.shape[3]

			if str(one_layer.lrn_param.local_size) == '':
				local_size = "5"
			else:
				local_size = str(one_layer.lrn_param.local_size)
			if str(one_layer.lrn_param.alpha) == '':
				alpha = "1"
			else:
				alpha = str(round(one_layer.lrn_param.alpha, 6))
			if str(one_layer.lrn_param.beta) == '':
				beta = "5"
			else:
				beta = str(round(one_layer.lrn_param.beta, 4))

			parameter_list = []
			parameter_list.append(float(alpha))
			parameter_list.append(float(beta))
			parameter_list = FpConvert.f32Tof16(parameter_list)

			shaves_used = 0
			if refeed_flag == True:
				shaves_used = layer_configuration['Shave'][id - concat_id - 1]
			else:
				shaves_used = layer_configuration[0]

			if one_layer.lrn_param.local_size == 5 and float(round(one_layer.lrn_param.alpha, 6)) == 0.0001 and float(round(one_layer.lrn_param.beta, 4)) == 0.75:
				network_list.append("    network_vector.emplace_back(new Lrn(" 
									+ "(u8*)(&branch_output_buffer_{0}_{1}{2})".format(str(local_counter), str(local_buffer_index), buffer_offset_str) + ", "
									+ str(bottom_vector[0]) + ", "
									+ str(shaves_used) + "));\n\n")
			else:
				network_list.append("    network_vector.emplace_back(new Lrn(" 
									+ "(u8*)(&branch_output_buffer_{0}_{1}{2})".format(str(local_counter), str(local_buffer_index), buffer_offset_str) + ", "
									+ str(bottom_vector[0]) + ", "
									+ str(shaves_used) + ", "
									+ local_size + ", " 
									+ str(parameter_list[0]) + ", " 
									+ str(parameter_list[1]) + "));\n\n")

			del parameter_list[:]
			branch_buffer_index = branch_buffer_index ^ 1
			network_defines_list[0] = network_defines_list[0].replace("LRN FALSE", "LRN TRUE")
			id += 1

		################################################################		CONCAT LAYER

		elif ''.join(map(str, one_layer.type)) == 'Concat':

			if network.blobs[str(one_layer.name)].data.shape[1]*network.blobs[str(one_layer.name)].data.shape[2]*network.blobs[str(one_layer.name)].data.shape[3] > biggest_blob_list[branch_counter][branch_buffer_index]:
				biggest_blob_list[branch_counter][branch_buffer_index] = network.blobs[str(one_layer.name)].data.shape[1]*network.blobs[str(one_layer.name)].data.shape[2]*network.blobs[str(one_layer.name)].data.shape[3]
			
			network_list.append("    network_vector.emplace_back(new Concat(" 
								+ "(u8*)(&branch_output_buffer_{0}_{1})".format(str(branch_counter), str(split_buffer_index)) + ", "
								+ str(network.blobs[str(one_layer.name)].data.shape[1]) + ", " 
								+ str(network.blobs[str(one_layer.name)].data.shape[2]) + ", " 
								+ str(network.blobs[str(one_layer.name)].data.shape[3]) + "));\n\n")
				
			branch_buffer_index = split_buffer_index ^ 1
			network_defines_list[0] = network_defines_list[0].replace("LINEAR TRUE", "LINEAR FALSE")
			id += 1
			concat_id += 1

		################################################################		RELU LAYER

		elif ''.join(map(str, one_layer.type)) == 'ReLU':
			relu_id += 1
			index_offset = 0
			if str(one_layer.bottom) == str(one_layer.top):
				for index, name in enumerate(layer_name_list):
					if name == ''.join(map(str, one_layer.top)).replace("/", "_").replace('-', '_'):
						index_offset = index
			else:
				for index, name in enumerate(layer_name_list):
					if name == ''.join(map(str, one_layer.bottom)).replace("/", "_").replace('-', '_'):
						index_offset = index
			network_list[index_offset + network_list_offset] = network_list[index_offset + network_list_offset].replace(", 0));", ", 1));")

		################################################################		DROPOUT LAYER

		elif ''.join(map(str, one_layer.type)) == 'Dropout':
			dropout_id += 1

	###---------------------------------------------telos tis for---------------------------------------------###


#############################################	network.cpp
	try:
		network_list.append("    return std::move(network_vector);\n}")
		network_file = open("./network.cpp", "w")
		for line in network_list:
			network_file.write(line)
		network_file.close()

	except IOError:
		print '\033[91m' + '[API] Could not write \'network.cpp\'. Exiting...' + '\033[0m'
		sys.exit(2)
#############################################	network.h
	try:
		network_lib_file = open("./network.h", "w")
		for line in network_lib_list:
			network_lib_file.write(line)
		network_lib_file.close()

	except IOError:
		print '\033[91m' + '[API] Could not write \'network.h\'. Exiting...' + '\033[0m'
		sys.exit(2)
####################################################	network_defines.h
	try:
		network_defines_file = open("./network_defines.h", "w")
		
		network_defines_file.write(network_defines_list[0])
		network_defines_file.close()

	except IOError:
		print '\033[91m' + '[API] Could not write \'network_defines.h\'. Exiting...' + '\033[0m'
		sys.exit(2)
#####################################################	weight_data.c
	branch_counter = 0
	weight_list.append("//------------------Output buffer declaration---------------------//\n")
	for item in biggest_blob_list:
		branch_buffer_index = 0
		for index in item:
			if index > 0:
				weight_list.append("fp16 DDR_BUFFER branch_output_buffer_{0}_{1}[{2}];\n".format(str(branch_counter), str(branch_buffer_index), str(index)))
			branch_buffer_index +=1
		branch_counter += 1
	weight_list.append("\n")

	try:
		weight_file = open("./weight_data.c", "w")
		for line in weight_list:
			weight_file.write(line)
		weight_file.write(weight_string)
		weight_file.close()

	except IOError:
		print '\033[91m' + '[API] Could not write \'weight_data.c\'. Exiting...' + '\033[0m'
		sys.exit(2)
##############################################	weight_data.h
	try:
		weight_lib_file = open("./weight_data.h", "w")

		branch_counter = 0
		weight_lib_list.append("//------------------Output buffer declaration---------------------//\n")
		for item in biggest_blob_list:
			branch_buffer_index = 0
			for index in item:
				if index > 0:
					weight_lib_list.append("extern u8 branch_output_buffer_{0}_{1}[];\n".format(str(branch_counter), str(branch_buffer_index)))
				branch_buffer_index +=1
			branch_counter += 1
		weight_lib_list.append("\n")

		for line in weight_lib_list:	
			weight_lib_file.write(line)

		weight_lib_file.write(weight_library_string + "\n#endif")
		weight_lib_file.close()

	except IOError:
		print '\033[91m' + '[API] Could not write \'weight_data.h\'. Exiting...' + '\033[0m'
		sys.exit(2)

	return
	###----------------------------------------------------end function----------------------------------------###

if __name__ == '__main__':

	#Argument initialization
	parser = argparse.ArgumentParser(description = 'Import pre-trained Neural Network and Dataset to Generate source code to execute on Myriad')
	Required_args = parser.add_argument_group('Required Arguments')
	Required_args.add_argument( '-p', '--prototxt', help = "Define path to prototxt model", required = True)
	Required_args.add_argument( '-c', '--caffemodel', help = "Define path to trained caffemodel", required = True)
	Optional_args = parser.add_argument_group('Optional Arguments')
	Optional_args.add_argument( '-i', '--image', default = '', help = "Define path to input image to import")
	Optional_args.add_argument( '-s', '--shaves', default = '10', help = "Define number of shaves to be used for the network (Default: 10)")
	Optional_args.add_argument( '-pr', '--profile', choices = ['high_res', 'low_res'], help = "Select for profile mode.\n'\high_res'\: high resolution sampling\n'\low_res'\: faster energy profiling")
	Optional_args.add_argument( '-cl', '--clean', action = 'store_true', help = "Select for clean compile")
	Optional_args.add_argument( '-sv', '--server', choices = ['keep_log', 'silent'], help = "Select for handling MoviDebug Server. keep_log provides an output file of the session")
	Optional_args.add_argument( '-an', '--analyse_only', help = "Select by providing path to myriad profile csv file for explicitely analyzing data")
	Optional_args.add_argument( '-re', '--refeed', choices = ['performance', 'efficiency'], help = "Select for refeeding optimal configurations to Myriad")
	args = parser.parse_args()

	if int(args.shaves) not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
		parser.error('\033[91m' + "Invalid number of shaves defined" + '\033[0m')
		sys.exit(1)

	if args.profile == None:
		profile_flag = False
	else:
		profile_flag = True

	if (not args.prototxt) and (not args.caffemodel):
		parser.error('\033[91m' + '--caffemodel is required in this mode' + '\033[0m') 
		sys.exit(1)

	prototxt_path = args.prototxt.split('/')
	for item in prototxt_path:
		network_name = item
	network_name = network_name.split('.')[0]

	if args.analyse_only:

		optimal_time_linear = {}
		optimal_energy_linear = {}
		optimal_parallel = []

		optimal_time_linear, optimal_energy_linear, optimal_parallel = network_exploration(args.analyse_only, args.prototxt, args.caffemodel, network_name)

		if args.refeed == 'performance': #todo for parallel
			library_generator(args.prototxt, args.caffemodel, args.image, optimal_time_linear, 0)
			if args.image == '':
				print '\033[93m' + '[API] WARNING: Input Image is not specified! Results of computation will be wrong!' + '\033[0m'
			compile_and_execute(args.clean, args.server, network_name, False) #todo one more arg dual_mode if optimal_parallel
		elif args.refeed == 'efficiency':
			library_generator(args.prototxt, args.caffemodel, args.image, optimal_energy_linear, 0)
			if args.image == '':
				print '\033[93m' + '[API] WARNING: Input Image is not specified! Results of computation will be wrong!' + '\033[0m'
			compile_and_execute(args.clean, args.server, network_name, False) #todo one more arg dual_mode if optimal_parallel
	else:

		if args.profile == 'high_res':
			library_generator(args.prototxt, args.caffemodel, args.image, [args.shaves], 50)
		elif args.profile == 'low_res':
			library_generator(args.prototxt, args.caffemodel, args.image, [args.shaves], 20)
		elif args.profile == None:
			library_generator(args.prototxt, args.caffemodel, args.image, [args.shaves], 0)
		if args.image == '':
			print '\033[93m' + '[API] WARNING: Input Image is not specified! Results of computation will be wrong!' + '\033[0m'

		# compile_and_execute(args.clean, args.server, network_name, profile_flag)
		if args.profile != None:
			optimal_time_linear = {}
			optimal_energy_linear = {}
			optimal_parallel = []

			optimal_time_linear, optimal_energy_linear, optimal_parallel = network_exploration('./{0}_profile.csv'.format(network_name), args.prototxt, args.caffemodel, network_name)

			if args.refeed == 'performance': #todo for parallel
				library_generator(args.prototxt, args.caffemodel, args.image, optimal_time_linear, 0)
				if args.image == '':
					print '\033[93m' + '[API] WARNING: Input Image is not specified! Results of computation will be wrong!' + '\033[0m'
				compile_and_execute(True, args.server, network_name, False) #todo one more arg dual_mode if optimal_parallel
			elif args.refeed == 'efficiency':
				library_generator(args.prototxt, args.caffemodel, args.image, optimal_energy_linear, 0)
				if args.image == '':
					print '\033[93m' + '[API] WARNING: Input Image is not specified! Results of computation will be wrong!' + '\033[0m'
				compile_and_execute(True, args.server, network_name, False) #todo one more arg dual_mode if optimal_parallel

	print '\033[92m' + '[API] Execution sucessful!\n[API] Exiting...' + '\033[0m'
	sys.exit(0)
