function R = vecMat(omega)
%-------------------------------------------------------------------------
% Return the rotation matrix equal to exp(omega)

    rn = norm (omega);
    if rn > 1.0e-12
        r = omega/rn;
        s = sin(rn); c = 1-cos(rn);

        srx = s*r(1); sry = s*r(2); srz = s*r(3);

        crx = c*r(1); cry = c*r(2); crz = c*r(3);

        crxrx = crx*r(1); cryry = cry*r(2); crzrz = crz*r(3);
        crxry = crx*r(2); cryrz = cry*r(3); crxrz = crx*r(3);

        R = [1-cryry-crzrz -srz+crxry sry+crxrz;...
            srz+crxry 1-crxrx-crzrz -srx+cryrz;...
            -sry+crxrz srx+cryrz 1-crxrx-cryry];
    else
        R = eye(3);
    end
end