function get_csv_data_steps(del_t,n_points,n_steps,snr,base_var,var_var,save_on_off)

eta = 1/base_var;nu = gamrnd(1/var_var,snr*var_var,[n_points,1]);%nu = (1/10)*[2:n_points+1]';%
var_n = (eta+nu) ./ (eta * nu);
heights = cumsum([0;snr*base_var*randn([n_steps,1])]); 
data = linspace(del_t(1),del_t(2),n_points)';
step_times =   sort(del_t(1) + (del_t(2)-del_t(1))*rand([n_steps,1])) ;
step_corr = sum(data > step_times',2)+1;
[del_t(1);step_times];

% data = [data,normrnd(zeros(size(var_n)),var_n) + heights(step_corr),var_n,heights(step_corr)];
data = [data,normrnd(zeros(size(var_n)),var_n) + heights(step_corr),nu,heights(step_corr)];

figure;
tiledlayout(2,1,'TileSpacing','tight','Padding','tight');
nexttile();
scatter(data(:,1),data(:,2));hold on
scatter(data(:,1),data(:,4));
hold off;legend('data','ground truth','Orientation','vertical','Box','off')
nexttile();
plot(data(:,1),data(:,3))

% if save_on_off
    writematrix(data(:,1:2),'../simple_data_no_var.csv');
    writematrix(data(:,1:3),'../simple_data_no_gt.csv');
    writematrix(data,'../my_data_gt.csv');
% end




end