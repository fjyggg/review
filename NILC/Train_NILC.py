from Algorithms.compared_algorithms.Learn_NILC import LearnNILC
import numpy as np
import pyLasaDataset as lasa
np.random.seed(5)


Type = 'Snake'
data = getattr(lasa.DataSet, Type)
demos = data.demos

d_x = np.shape(demos[0].pos)[0]
# Width of the Neural networks
d_H = 100
print('--- Start training ---')
save_paths = ['NilcParameters/Nilc_weight_parameter_for_' + Type + '_dH' + str(d_H) + '.txt',
              'NilcParameters/Nilc_feature_parameter_for_' + Type + '_dH' + str(d_H) + '.txt']

'''
nilc_weight_parameters = np.loadtxt(save_paths[0])
nilc_feature_parameters = np.loadtxt(save_paths[1])
A = nilc_feature_parameters[0: (d_x * d_H)].reshape(d_H, d_x)
b = nilc_feature_parameters[(d_x * d_H):(d_x * d_H + d_H)]
'''

# If you want train the parameters, set A=none, b=None
nilc_learner = LearnNILC(demos=demos, d_H=d_H, A=None, b=None)
nilc_weight_parameters = nilc_learner.train(save_paths)
print('--- Training successful ---')

print('plotting results ...')
nilc_learner.show_learning_result(nilc_weight_parameters, num_levels=10)
E_1, std_E1, E_2, std_E2 = nilc_learner.compute_evaluations(nilc_weight_parameters)
print((E_1, std_E1, E_2, std_E2))