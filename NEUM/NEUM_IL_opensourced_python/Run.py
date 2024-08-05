'''
A test file for the approach proposed in the paper:
"Learning a flexible neural energy function with a unique minimum
for stable and accurate demonstration learning"
'''

import numpy as np
import pyLasaDataset as lasa
from Algorithms.Learn_NEUM import LearnNeum
from Algorithms.Learn_GPR_ODS import LearnOds
from Algorithms.Learn_SDS import LearnSds
np.random.seed(5)

Type = 'BendedLine'
data = getattr(lasa.DataSet, Type)
demos = data.demos


def construct_demonstration_set(demos, start=1, end=-1, gap=5, used_tras=[1, 2, 3, 4, 5, 6]):
    n_tra = len(used_tras)
    x_set = []
    dot_x_set = []
    t_set = []
    for i in range(n_tra):
        x_set.append(demos[used_tras[i]].pos[:, start:end:gap].T)
        dot_x_set.append(demos[used_tras[i]].vel[:, start:end:gap].T)
        t_set.append(demos[used_tras[i]].t[0, start:end:gap])

    x_set = np.array(x_set)
    dot_x_set = np.array(dot_x_set)
    t_set = np.array(t_set)
    return x_set, dot_x_set, t_set


# ---------- Learning (Loading) the neural energy function NEUM ------------
d_H = 10
manually_design_set_neum = construct_demonstration_set(demos, start=1, end=-1, gap=20)
neum_learner = LearnNeum(manually_design_set=manually_design_set_neum, d_H=d_H, L_2=1e-6)
print('--- Start energy function training (loading)---')
beta = 1.0
save_path = 'NeumParameters/Neum_parameter_for_' + Type + '_beta' + str(beta) + '_dH' + str(d_H) + '.txt'
# Training or Loading
# neum_parameters = neum_learner.train(save_path=save_path, beta=beta, maxiter=1000)
neum_parameters = np.loadtxt(save_path)
print('--- Training (Loading) completed ---')
print('plotting energy function learning results ...')
neum_learner.show_learning_result(neum_parameters, num_levels=10)
print('Plotting finished')

# ------------------- Learning (Loading) original ADS --------------------
observation_noise = None
gamma_oads = 0.5
manually_design_set_oads = construct_demonstration_set(demos, start=40, end=-1, gap=5)
ods_learner = LearnOds(manually_design_set=manually_design_set_oads, observation_noise=observation_noise, gamma=gamma_oads)
print('--- Start original ads training (loading) ---')
save_path = 'OadsParameters/Oads_parameter_for_' + Type + '.txt'
# Training or Loading. Using "ods_learner.set_param" when loading parameters
# ods_learner.train(save_path)
oads_parameters = np.loadtxt(save_path)
ods_learner.set_param(oads_parameters)
print('--- Training (Loading) completed ---')

# ------------------- Formulate the stable ADS ----------------------
print('Formulating stable ADS ...')
sds_learner = LearnSds(lf_learner=neum_learner, ods_learner=ods_learner)


def func_rho(x):
    '''
    Define function rho(x),
    see Eq.(57) in the paper
    '''
    gamma = np.max(np.sqrt(np.sum(neum_learner.dot_x_set ** 2, axis=1))) / 1e3
    dvdx = neum_learner.dvdx(neum_parameters, x)
    return np.sqrt(np.dot(dvdx, dvdx)) * gamma


# ---------- Run the stable ADS ------------
# Position constraint setting
P = np.array([[1.0, 0.0], [0.0, 6.0]])
r_thres = 0.0 # r_thres = 0.0 implies that the position constraint will not be used
eta = 0.05
print('plotting stable ADS results ...')
sds_learner.show_learning_result(lf_parameter=neum_parameters, func_rho=func_rho, P=P, r_thres=r_thres, eta=eta)
print('Plotting finished')



