function [q, omega, domega, t] = computeOmega(q,dt)

N = length(q);
t = linspace(dt, N*dt, N);


% Normalize quaternions
for i = 1:N
    tmp = quat_norm(array2quat(q(i,:)'));
    q(i,:) = q(i,:)/tmp;
end

% Calculate derivatives
for j = 1:4
    dqq(:,j) = gradient(q(:,j), t);
end
% dqq(1,:) = [0 0 0 0];
% dqq(end,:) = [0 0 0 0];
% Calculate omega and domega
for i = 1:N
    dq.s = dqq(i,1);
    for j = 1:3
        dq.v(j,1) = dqq(i,j+1);
    end
    omega_q = quat_mult(dq, quat_conjugate(array2quat(q(i,:)')));
    omega(i,:) = 2*omega_q.v;
end
for j = 1:3
    domega(:,j) = gradient(omega(:,j), t);
end

omega(1,:) = [0; 0; 0];
omega(N,:) = [0; 0; 0];
domega(1,:) = [0; 0; 0];
domega(N,:) = [0; 0; 0];

end