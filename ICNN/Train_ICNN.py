from Algorithms.compared_algorithms.Learn_ICNN import LearnIcnn
import numpy as np
import pyLasaDataset as lasa
np.random.seed(5)

Type = 'BendedLine'
data = getattr(lasa.DataSet, Type)
demos = data.demos

d_H = np.array([2, 100, 100, 1])
icnn_learner = LearnIcnn(demos=demos, d_H=d_H)
print('--- Start training ---')
beta = 1.0
save_path = 'IcnnParameters/Icnn_parameter_for_' + Type + '_beta' + str(beta) + '_d_H' + str(d_H) + '.txt'

icnn_parameters = icnn_learner.train(save_path=save_path, beta=beta)
# icnn_parameters = np.loadtxt(save_path)
print('--- Training successful ---')

print('plotting results ...')
icnn_learner.show_learning_result(icnn_parameters, num_levels=10)
E_1, std_E1, E_2, std_E2 = icnn_learner.compute_evaluations(icnn_parameters)
print((E_1, std_E1, E_2, std_E2))
