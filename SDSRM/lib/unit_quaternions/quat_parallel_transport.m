%% Parallel transportation a vector w from the tangent space of p t q  
function wt = quat_parallel_transport(w, p, q)
	v = [0; quat_log(p, q)];
    nv = norm(v);
	if nv < 1e-10
		disp('Angle of rotation too small (<1e-10)');
		wt = w;
		return;
	end
	uv = v / nv;
    mv = (v * v.') / (nv)^2;
    
    pv = quat2array(p);
    w = [0; w];
	w4 = (-pv*sin(nv) + uv*cos(nv))*(v.'*w) + (eye(4)-mv)*w;	
    wt = w4(2:4);
end



