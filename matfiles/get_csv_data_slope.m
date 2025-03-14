function get_csv_data_slope(del_t,n_points,n_steps,snr,base_var,var_var,save_on_off)

eta = 1/base_var;nu = gamrnd(1/var_var,snr*var_var,[n_points,1]);%nu = (1/10)*[2:n_points+1]';%
if isfinite(eta)
    var_n = (eta+nu) ./ (eta * nu);
else
    var_n = 1 ./ nu;
end

heights = snr*max(1,base_var)*ones(n_steps,1); 
data = linspace(del_t(1),del_t(2),n_points)';
step_times = [20; 24];
% step_times =   sort(del_t(1) + 0.5*(del_t(2)-del_t(1))*rand([n_steps,1])) ;

impulses = [];
for s = 1:n_steps
    impulses = [impulses,slope_kernel(data-step_times(s))];
end


[del_t(1);step_times];

% data = [data,normrnd(zeros(size(var_n)),var_n) + heights(step_corr),var_n,heights(step_corr)];
data = [data,normrnd(zeros(size(var_n)),var_n) + sum(heights' .* impulses,2),nu,sum(heights' .* impulses,2)];

figure;
tiledlayout(2,1,'TileSpacing','tight','Padding','tight');
nexttile();
scatter(data(:,1),data(:,2));hold on
scatter(data(:,1),data(:,4));
hold off;legend('data','ground truth','Orientation','vertical','Box','off')
nexttile();
plot(data(:,1),data(:,3))

if save_on_off
    writematrix(data(:,1:2),'../simple_data_no_var.csv');
    writematrix(data(:,1:3),'../simple_data_no_gt.csv');
    writematrix(data,'../my_data_gt.csv');
end




end