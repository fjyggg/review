function y = quat2array(q)

y = zeros(4,1);
y(1,1) = q.s;
y(2:4,1) = q.v;

end