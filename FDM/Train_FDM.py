from 研前期.compare_algorithm.FDM.Fast_diffeomorphic_matching import FastDiffMatching
import numpy as np
import pyLasaDataset as lasa
np.random.seed(5)

Type = 'Snake'
data = getattr(lasa.DataSet, Type)
demos = data.demos

K = 150
mu = 0.9
beta = 0.5  #这三个参数和文中给的是一样的
fdm_learner = FastDiffMatching(demos=demos, K=K, beta=beta, mu=mu)
print('--- Start training ---')
save_path = 'FdmParameters/parameter_for_' + Type + '_K' + str(K) + '_mu' + str(mu) + '_beta' + str(beta) + '.txt'
# fdm_parameters = fdm_learner.train(save_path=save_path)
fdm_parameters = np.loadtxt(save_path)
print('--- Training successful ---')

# print(fdm_learner.Phi(np.zeros(2), fdm_parameters))
print('plotting results ...')
violated_numbers = fdm_learner.plot_lf_learning_result(fdm_parameters, num_levels=10)
# E_1, std_E1, E_2, std_E2 = fdm_learner.compute_evaluations(fdm_parameters, beta=10.0)
# print((E_1, std_E1, E_2, std_E2))