%GENERATE_ORIENTATION_DATA generate quaternion trajectory, angular velocities and
%   accelerations between 2 given quaternions using minimum jerk algorithm.
%
% INPUTS:
%   -q1:  initial quaternion
%   -q2:  final quaternion
%   -tau: duration time
%   -dt:  sample time
%
% OUTPUTS:
%   -q:      quaternion trajectory
%   -omega:  angular velocity profile
%   -dmoega: angular acceleration profile
%   -t:      time vector
%

% Copyright (C) 2013-2018, by Abu-Dakka, Fares J.
%
% This file is part of The Cartesian Dynamic Movement Preimitives Library
% (CartDMPL)
%
% CartDMPL is free software: you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as published
% by the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% CartDMPL is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Lesser General Public License for more details.
%
% You should have received a copy of the GNU Leser General Public License
% along with CartDMPL.  If not, see <http://www.gnu.org/licenses/>.
function [q, omega, domega, t] = generate_orientation_data(q1, q2, tau, dt, as_array)

if(nargin < 5)
    as_array = 0;
end

N = round(tau / dt) + 1;
t = linspace(0, tau, N);

quat_temp = quat_structure;
q = repmat(quat_temp,N,1);
qq = zeros(4,N);
dqq = zeros(4,N);
omega = zeros(3,N);
domega = zeros(3,N);

% Generate spline data from q1 to q2
a = minimum_jerk_spline(q1.s, 0, 0, q2.s, 0, 0, tau);
for i = 1:N
    q(i).s = minimum_jerk(t(i), a);
end
for j = 1:3
    a = minimum_jerk_spline(q1.v(j), 0, 0, q2.v(j), 0, 0, tau);
    for i = 1:N
        q(i).v(j,1) = minimum_jerk(t(i), a);
    end
end

% Normalize quaternions
for i = 1:N
    tmp = quat_norm(q(i));
    q(i).s = q(i).s / tmp;
    q(i).v = q(i).v / tmp;
    qq(:,i) = [q(i).s; q(i).v];
end

% Calculate derivatives
for j = 1:4
    dqq(j,:) = gradient(qq(j,:), t);
end

% Calculate omega and domega
for i = 1:N
    dq.s = dqq(1,i);
    for j = 1:3
        dq.v(j,1) = dqq(j+1,i);
    end
    omega_q = quat_mult(dq, quat_conjugate(q(i)));
    omega(:,i) = 2*omega_q.v;
end
for j = 1:3
    domega(j,:) = gradient(omega(j,:), t);
end

omega(:,1) = [0; 0; 0];
omega(:,N) = [0; 0; 0];
domega(:,1) = [0; 0; 0];
domega(:,N) = [0; 0; 0];

if(as_array)
    q = qq;
end
end
