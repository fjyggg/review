function [q, perturbation] = quat_perturbation(q, size)
%-------------------------------------------------------------------------
% Calculates difference vector between 2 quaternions
    perturbation = size * (rand(1,4) - 0.5);
    q.s = q.s + size * perturbation(1);
    q.v = q.v + size * perturbation(2:4)';
    tmp = quat_norm(q);
    q.s = q.s / tmp;
    q.v = q.v / tmp;
end