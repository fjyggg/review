clear;
close all;

%% Add GMR lib and shapes from Riemannian LASA dataset
addpath(genpath('lib'))
addpath(genpath('data'))

%% Load training data
LASA_shape_number = 3;
% Load demonstrations
demos = load_LASA_UQ_models('LASA_HandWriting_UQ/', LASA_shape_number);
q_goal = array2quat(demos{1}.quat(:,end));
    
% Downsample training data (faster training)     
sampling_rate = 10;
N = size(demos{1}.quat, 2);
sam_ind = [1:sampling_rate:N N];
demo_len = length(sam_ind);
q_ts_demo = []; 
q_ts_vel = []; 
q_init_all = [];
q_ts_base = [];
q_ts_demo_init_all = [];
for i=1:length(demos)
    dt_ = sampling_rate*demos{i}.dt;

    q_init_all = [q_init_all, demos{i}.quat(:,1)];
    q_demo_array{i} = demos{i}.quat(:, sam_ind);
    q_ts_demo_init_all = [q_ts_demo_init_all, demos{i}.tsPos(:, 1)];
    q_ts_demo = [q_ts_demo, demos{i}.tsPos(:, sam_ind)];
    q_ts_vel = [q_ts_vel [diff(q_ts_demo, [], 2)./dt_, zeros(3,1)]];

    % Base stable dynamics
    q_curr = array2quat(demos{i}.quat(:,1));

    % Compute control to have zero error after 'demo_len' steps
    kq = demo_len * dt_ / 5;
    for d=1:demo_len    
        log_q = quat_log(q_curr, q_goal);

        d_log_q = - kq * log_q;

        q_ts_base = [q_ts_base, log_q];

        log_q = log_q + d_log_q * dt_;

        % Convert to Riemmanian Manifold
        tmp = quat_exp(log_q);
        q_curr = quat_mult(tmp, q_goal);
    end
end

%% Learn a diffeomorphism between tangent spaces
D = 3;
Data = [q_ts_base; q_ts_demo];
in = 1:D;
out = D+1:2*D; 

nbStates = 10;

[Priors, Mu, Sigma] = EM_init_kmeans(Data, nbStates);
Mu(:,1) = [mean(q_ts_demo_init_all, 2); mean(q_ts_demo_init_all, 2)];
Mu(:,end) = Data(:,end);
[Priors, Mu, Sigma] = EM(Data, Priors, Mu, Sigma);   
Mu(:,end) = Data(:,end);

fun_jac = @(pt) GMR_with_Jacobian(Priors, Mu, Sigma, pt, in, out);

% Simulation 
for d=1:length(demos)
    q_curr = array2quat(q_init_all(:,d));
    q_diff_array = q_init_all(:,d);
    quat_err = quat_log(q_curr, array2quat(demos{d}.quat(:, 1)));
    q_ts_des = demos{d}.tsPos(:, sam_ind);

    for i=1:demo_len-1
        % Project to tangent space 
        q_ts_curr = quat_log(q_curr, q_goal);

        % Apply diffeomorphism and Compute Jacobian
        [q_ts_diff, Jac] = fun_jac(q_ts_curr);

        % Stable dynamics
        omega_diff = - kq  * Jac * q_ts_curr;

        % Integration
        q_ts_diff = q_ts_diff + dt_ * omega_diff;

        % Convert to Riemmanian Manifold
        tmp = quat_exp(q_ts_diff);
        q_diff = quat_mult(tmp, q_goal);
    
        % Stable dynamics
        omega = - kq * q_ts_curr;
        % Integration
        q_ts_curr = q_ts_curr + dt_ * omega;
    
        tmp = quat_exp(q_ts_curr);
        q_curr = quat_mult(tmp, q_goal);
            
        % Save data 
        q_diff_array = [q_diff_array, quat2array(q_diff)];
            
    end
        q_diff_save{d} = q_diff_array;

    %% Plot Results
    c_green = [0 127 0]/255;
    c_brown = [212 85 0]/255;
    c_red   = [170 0 0]/255;
    c_blue  = [0 113 188]/255;
    c_yello = [236 176 31]/255;
    time_ = dt_*[0:demo_len-1];
    %figure(modIt)
    h1 = figure(1);
    plot(time_, q_diff_save{d}(1,:), 'color', c_yello, 'LineWidth', 3)
    hold on
    plot(time_, q_diff_save{d}(2,:), 'color', c_blue, 'LineWidth', 3)
    plot(time_, q_diff_save{d}(3,:), 'color', c_green, 'LineWidth', 3)
    plot(time_, q_diff_save{d}(4,:), 'color', c_brown, 'LineWidth', 3)

    plot(time_, q_demo_array{d}', 'k--', 'LineWidth', 3)
       
end

axis([0 3 -0.55 1.1])
axis square;
set(gca, 'XTick', [0 1.5 3], 'YTick', [-0.5 0 0.5 1], 'FontName','Times','Fontsize',24);
xlabel('Time [s]', 'interpreter', 'latex', 'FontName','Times','Fontsize',24);
ylabel('$\mathbf{\Omega}$', 'interpreter', 'latex', 'FontName','Times','Fontsize',24);
title('Unit quaternion','FontName','Times','Fontsize',24);
set(gca, 'FontName','Times','Fontsize',24);

%print(h1, ['./Results/' num2str(demo_iter) '_UQ_GMM.pdf'], '-dpdf','-r300')

% Arcosine re-defitinion to make sure the distance between antipodal quaternions is zero (2.50 from Dubbelman's Thesis)
function acosx = acoslog(x)
	for n=1:size(x,2)
		% sometimes abs(x) is not exactly 1.0
		if(x(n)>=1.0)
			x(n) = 1.0;
		end
		if(x(n)<=-1.0)
			x(n) = -1.0;
		end
		if(x(n)>=-1.0 && x(n)<0)
			acosx(n) = acos(x(n))-pi;
		else
			acosx(n) = acos(x(n));
		end
	end
end