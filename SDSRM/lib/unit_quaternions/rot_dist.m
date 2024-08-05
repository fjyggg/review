function distance = rot_dist(R1, R2)
%-------------------------------------------------------------------------
dR = rot_log(R2*R1');
distance = norm(dR);
end