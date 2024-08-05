function log_R = rot_log(R)
%-------------------------------------------------------------------------
% Calculates logarithm of orientation difference between quaternions
do = vrrotmat2vec(R);
log_R = do(4)*do(1:3)';
end