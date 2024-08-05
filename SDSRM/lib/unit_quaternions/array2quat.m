function q = array2quat(y)

q = struct('s',cell(1),'v',cell(1));
q.s = y(1);
q.v = y(2:4);

end