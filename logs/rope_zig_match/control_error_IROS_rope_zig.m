% close all
clear all

space_hold_1 = '%n';
space_hold_2 = '%n %n';

%%%%%%%%%%% Error Analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file_error_1 = 'controll_error/control_error_realtime.txt';
file_t_1 = 'controll_error/control_time.txt';

parameter_set_dd_wo = 'e^r_B';
parameter_set_mm = 'e^r_N';

[mean_error_mm, mean_error_dd_wo]  = textread(file_error_1, space_hold_2,'headerlines',1);

min_error_of_all_controllers = min(mean_error_mm, mean_error_dd_wo);

[t_1] = textread(file_t_1, space_hold_1, 'headerlines',1);
t_dd_wo = t_1;

t_mm = t_1;
fig_path_integral_relative_error = 'controll_error/integral_relative_error_MM.png';

relative_error_mm = mean_error_mm - min_error_of_all_controllers;
relative_error_dd_wo = mean_error_dd_wo - min_error_of_all_controllers;

relative_sum_mm = cumsum(relative_error_mm);
relative_sum_dd_wo = cumsum(relative_error_dd_wo);

show_ind = 1:length(t_mm);

fig_1 = figure(2);
plot(t_mm(show_ind), relative_sum_mm(show_ind),...
        t_dd_wo(show_ind), relative_sum_dd_wo(show_ind),'LineWidth',2)
legend(parameter_set_mm, parameter_set_dd_wo, 'Location', 'northwest')
title('Integration of relative control error')
xlabel('time (s)')
ylabel('intergral of e^r')
saveas(fig_1, fig_path_integral_relative_error)






