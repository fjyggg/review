%QUAT_NORM multiply two quaternions.

% INPUTS:
%   -q: quaternion

% OUTPUTS:
%   -res: the norm of quaternion q

% References: If some parts of the code were useful for your research,
%             please cite the following related publications used to
%             construct this code. Moreover, please consider making your 
%             own research available in this way.
% [1] @article{AbuDakka2015,
%           title={Adaptation of manipulation skills in physical contact with the environment to reference force profiles},
%           author={Abu-Dakka, Fares J. and Nemec, Bojan and J{\o}rgensen, Jimmy A. and Savarimuthu, Thiusius R. and Kr{\"u}ger, Norbert and Ude, Ale{\v{s}}},
%           journal={AUTON ROBOT},
%           volume={39},
%           number={2},
%           pages={199--217},
%           year={2015}
%     }
% [2] @inproceedings{ude2014orientation,
%           title={Orientation in cartesian space dynamic movement primitives},
%           author={Ude, Ale{\v{s}} and Nemec, Bojan and Petri{\'c}, Tadej and Morimoto, Jun},
%           booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
%           pages={2997--3004},
%           year={2014},
%           organization={IEEE}
%     }

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

function res = quat_norm(q)

res = norm([q.s; q.v]);

end
