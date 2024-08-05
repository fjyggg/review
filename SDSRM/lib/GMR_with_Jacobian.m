function [y, J_y, Sigma_y, beta] = GMR_with_Jacobian(Priors, Mu, Sigma, x, in, out)
%
% This function computes mean and variance of a Gaussian Mixture 
% Regression (GMR) and the Jacobian (Jx(y) = dy/dx) of y = E[p(y|x)].
% Beign the joint probability p(y,x) estimated with a Gaussian Mixture 
% Model (GMM), y, Sigma, and Jx(y) admit a closed-form expression.
%
% Inputs -----------------------------------------------------------------
%   o Priors:  1 x K array representing the prior probabilities of the K GMM 
%              components.
%   o Mu:      D x K array representing the centers of the K GMM components.
%   o Sigma:   D x D x K array representing the covariance matrices of the 
%              K GMM components.
%   o x:       P x N array representing N datapoints of P dimensions.
%   o in:      1 x P array representing the dimensions to consider as
%              inputs.
%   o out:     1 x Q array representing the dimensions to consider as
%              outputs (D=P+Q).
% Outputs ----------------------------------------------------------------
%   o y:       Q x N array representing the retrieved N datapoints of 
%              Q dimensions, i.e. expected means.
%   o J_y:     Q x Q x N array representing the derivative of the expected
%              means wrt the input vector x.
%   o Sigma_y: Q x Q x N array representing the N expected covariance 
%              matrices retrieved. 
%
% Copyright (c) 2006 Sylvain Calinon, LASA Lab, EPFL, CH-1015 Lausanne,
%               Switzerland, http://lasa.epfl.ch
%
% The program is free for non-commercial academic use. 
% Please contact the authors if you are interested in using the 
% software for commercial purposes. The software must not be modified or 
% distributed without prior permission of the authors.
% Please acknowledge the authors in any academic publications that have 
% made use of this code or part of it. Please use this BibTex reference: 
% 
% @article{Calinon06SMC,
%   title="On Learning, Representing and Generalizing a Task in a Humanoid 
%     Robot",
%   author="S. Calinon and F. Guenter and A. Billard",
%   journal="IEEE Transactions on Systems, Man and Cybernetics, Part B. 
%     Special issue on robot learning by observation, demonstration and 
%     imitation",
%   year="2006",
%   volume="36",
%   number="5"
% }

nbData = size(x,2);
nbVar  = length(out);
nbStates = size(Sigma,3);

%% Fast matrix computation (see the commented code for a version involving 
%% one-by-one computation, which is easier to understand).
%%
%% Compute the influence of each GMM component, given input x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Pxi = zeros(nbData, nbStates);
for i=1:nbStates
  Pxi(:,i) = Priors(i).*gaussPDF(x, Mu(in,i), Sigma(in,in,i));
end
beta = Pxi./repmat(sum(Pxi,2)+realmin,1,nbStates);

% ind = find(sum(beta,2) == 0);
% if ~isempty(ind)
%     for i=1:nbStates
%         tmp = x(:,ind)' - repmat(Mu(in,i)',length(ind),1);
%         score(i,:) = Priors(i)*(sum((tmp/Sigma(in,in,i)).*tmp, 2));
%     end
%     [i i]=min(score);
%     beta=beta';
%     beta(nbStates.*(ind-1)+i')=1;
%     beta=beta';
% end
%% Compute expected means y, given input x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_tmp = zeros(nbVar,nbData,nbStates);
x_mu  = zeros(nbVar,nbData,nbStates);
s_k_a = zeros(nbVar,nbVar,1,nbStates);
sigma_in_x_mu = zeros(nbVar,nbData,nbStates);
for j=1:nbStates
  x_mu(:,:,j) = (x-repmat(Mu(in,j),1,nbData));  
  s_k_a(:,:,1,j) = Sigma(out,in,j)/(Sigma(in,in,j));
  sigma_in_x_mu(:,:,j) = Sigma(in,in,j) \ x_mu(:,:,j);
  y_tmp(:,:,j) = repmat(Mu(out,j),1,nbData) + s_k_a(:,:,1,j) * x_mu(:,:,j);
end
beta_tmp = reshape(beta,[1 size(beta)]);
y_tmp2 = repmat(beta_tmp,[length(out) 1 1]) .* y_tmp;
y = sum(y_tmp2,3) + 1e-3*x;
%% Compute Jacobian (J_y = dy/dx), given input x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sum_h_sigma = sum(repmat(beta_tmp,[length(out) 1 1]) .* sigma_in_x_mu, 3);
s_dot_ky_tmp = zeros(nbVar,nbVar,nbData,nbStates);
for j=1:nbStates
    sum_tmp = (sum_h_sigma - sigma_in_x_mu(:,:,j));
    for i=1:nbData
        s_dot_ky_tmp(:,:,i,j) = sum_tmp(:,i) * y_tmp(:,i,j).';
    end
end
s_dot_ky_tmp = s_dot_ky_tmp + repmat(s_k_a, [1 1 nbData 1]);
beta_tmp = reshape(beta,[1 1 size(beta)]);
s_dot_ky_tmp = repmat(beta_tmp, [length(out) length(out) 1 1]) .* s_dot_ky_tmp;
J_y = sum(s_dot_ky_tmp,4);
J_y = J_y + sign(det(J_y))*1e-3;
%% Compute expected covariance matrices Sigma_y, given input x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargout > 2
    for j=1:nbStates
        Sigma_y_tmp(:,:,1,j) = Sigma(out,out,j) - (Sigma(out,in,j)/(Sigma(in,in,j))*Sigma(in,out,j));
    end
    beta_tmp = reshape(beta,[1 1 size(beta)]);
    Sigma_y_tmp2 = repmat(beta_tmp.*beta_tmp, [length(out) length(out) 1 1]) .* repmat(Sigma_y_tmp,[1 1 nbData 1]);
    Sigma_y = sum(Sigma_y_tmp2,4);
end


% %% Slow one-by-one computation (better suited to understand the algorithm) 
% %%
% %% Compute the influence of each GMM component, given input x
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i=1:nbStates
%   Pxi(:,i) = gaussPDF(x, Mu(in,i), Sigma(in,in,i));
% end
% beta = (Pxi./repmat(sum(Pxi,2)+realmin,1,nbStates))';
% %% Compute expected output distribution, given input x
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% y = zeros(length(out), nbData);
% Sigma_y = zeros(length(out), length(out), nbData);
% for i=1:nbData
%   % Compute expected means y, given input x
%   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   for j=1:nbStates
%     yj_tmp = Mu(out,j) + Sigma(out,in,j)*inv(Sigma(in,in,j)) * (x(:,i)-Mu(in,j));
%     y(:,i) = y(:,i) + beta(j,i).*yj_tmp;
%   end
%   % Compute expected covariance matrices Sigma_y, given input x
%   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   for j=1:nbStates
%     Sigmaj_y_tmp = Sigma(out,out,j) - (Sigma(out,in,j)*inv(Sigma(in,in,j))*Sigma(in,out,j));
%     Sigma_y(:,:,i) = Sigma_y(:,:,i) + beta(j,i)^2.* Sigmaj_y_tmp;
%   end
% end
