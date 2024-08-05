function [R, omega, domega] = convert_data(q, t)
%-------------------------------------------------------------------------
% Calculates logarithm of orientation difference between quaternions
    N = numel(t);
    R = cell(N);
    RR = zeros(9,N);
    dR = zeros(9,N);
    omega = zeros(3,N);
    domega = zeros(3,N);

    for i = 1:N
        R{i} = quat_mat(q(i));
        RR(:,i) = reshape(R{i}, 9, 1);
    end

    for i = 1:9
        dR(i,:) = gradient(RR(i,:), t);
    end
    for i = 1:N
        dRi = reshape(dR(:,i),3,3);
        Omega = dRi*R{i}';
        omega(:,i) = [Omega(3,2); Omega(1,3); Omega(2,1)];
    end

    for j = 1:3
        domega(j,:) = gradient(omega(j,:), t);
    end

    omega(:,1) = [0; 0; 0];
    omega(:,N) = [0; 0; 0];
    domega(:,1) = [0; 0; 0];
    domega(:,N) = [0; 0; 0];
end