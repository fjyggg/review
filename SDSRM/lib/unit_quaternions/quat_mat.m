function R = quat_mat(q)
%-------------------------------------------------------------------------
% Return the 3x3 rotation matrix described by a quaternion

    a = q.s^2; b = q.v(1)^2; c = q.v(2)^2; d = q.v(3)^2;
    e = 2*q.v(1)*q.v(2); f = 2*q.s*q.v(3);
    g = 2*q.v(1)*q.v(3); h = 2*q.s*q.v(2);
    i = 2*q.v(2)*q.v(3); j = 2*q.s*q.v(1);

    R = [a+b-c-d e-f     g+h; ...
        e+f     a-b+c-d i-j; ...
        g-h     i+j     a-b-c+d];
end
