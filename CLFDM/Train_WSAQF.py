from Algorithms.compared_algorithms.Learn_WSAQF import Learn_WSAQF
import numpy as np
import pyLasaDataset as lasa
np.random.seed(5)

Type = 'Snake'
data = getattr(lasa.DataSet, Type)
demos = data.demos

K = 3
wsaqf_learner = Learn_WSAQF(demos=demos, K=K, L2=1e-6)
print('--- Start training ---')
beta = 10.0
# save_path = 'WsaqfParameters/Wsaqf_parameter_for_' + Type + '_beta' + str(beta) + '_K' + str(K) + '.txt'
save_path = '../plot_paper_figs/fig13_data/WSAQF_parameters/Wsaqf_parameter_for_' + Type + '_beta' + str(beta) + '_K' + str(K) + '.txt'
wsaqf_parameters = wsaqf_learner.train(save_path=save_path, beta=beta, maxiter=500)
# wsaqf_parameters = np.loadtxt(save_path)
print('--- Training successful ---')

print('plotting results ...')
wsaqf_learner.show_learning_result(wsaqf_parameters, num_levels=10)
# loss_tra = np.loadtxt(save_path)