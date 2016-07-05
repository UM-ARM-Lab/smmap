%%
clear; clc;
fig = figure(5);
clf;
set( fig, 'PaperPositionMode', 'auto',...
          'Units', 'inches', ...
          'Position', [0 0 10 10] );

% draw_alpha = true;
draw_alpha = false;
plane_alpha = 0.1;

% y_curr = [0 0 0]';
% y_curr = [6.32551  6.34033 -1.05707]';
y_curr = [4.86121  4.98856 -0.82094]';
y_des = [10 10 10]';
num_arms = 2;

h_yCurr = plot3(y_curr(1), y_curr(2), y_curr(3), 'r+', 'MarkerSize', 20);
xlabel('x');
ylabel('y');
zlabel('z');
% axis([-1 12 -1 12 -1 12])
hold on
h_yDes = plot3(y_des(1), y_des(2), y_des(3), 'k+', 'MarkerSize', 20);

J_true = [
     1.06017     -0.0419678
    -0.0797539    1.05166
    -0.0927115   -0.073844
];

J_true_plane_corners = zeros(4,3);
J_true_plane_corners(1,:) = J_true * [-10; -10] + y_curr;
J_true_plane_corners(2,:) = J_true * [ 25; -10] + y_curr;
J_true_plane_corners(3,:) = J_true * [ 25;  25] + y_curr;
J_true_plane_corners(4,:) = J_true * [-10;  25] + y_curr;
if draw_alpha
    h_JTruePlane = patch(J_true_plane_corners(:, 1), J_true_plane_corners(:, 2), J_true_plane_corners(:, 3), 'c', 'FaceAlpha', plane_alpha);
else
    h_JTruePlane = patch(J_true_plane_corners(:, 1), J_true_plane_corners(:, 2), J_true_plane_corners(:, 3), 'c');
end

J1 = [
       1.05198     -0.039885
      -0.0643178    1.07534
      -0.106667    -0.0588499
 ];
J1_plane_corners = zeros(4,3);
J1_plane_corners(1,:) = J1 * [-10; -10] + y_curr;
J1_plane_corners(2,:) = J1 * [ 25; -10] + y_curr;
J1_plane_corners(3,:) = J1 * [ 25;  25] + y_curr;
J1_plane_corners(4,:) = J1 * [-10;  25] + y_curr;
if draw_alpha
    h_J1Plane = patch(J1_plane_corners(:, 1), J1_plane_corners(:, 2), J1_plane_corners(:, 3), 'm', 'FaceAlpha', plane_alpha);
else
    h_J1Plane = patch(J1_plane_corners(:, 1), J1_plane_corners(:, 2), J1_plane_corners(:, 3), 'm');
end

J2 = [
       1.07552       -0.058516
      -0.0664411      1.076
      -0.101702      -0.0813672
 ];
J2_plane_corners = zeros(4,3);
J2_plane_corners(1,:) = J2 * [-10; -10] + y_curr;
J2_plane_corners(2,:) = J2 * [ 25; -10] + y_curr;
J2_plane_corners(3,:) = J2 * [ 25;  25] + y_curr;
J2_plane_corners(4,:) = J2 * [-10;  25] + y_curr;
if draw_alpha
    h_J2Plane = patch(J2_plane_corners(:, 1), J2_plane_corners(:, 2), J2_plane_corners(:, 3), 'g', 'FaceAlpha', plane_alpha);
else
    h_J2Plane = patch(J2_plane_corners(:, 1), J2_plane_corners(:, 2), J2_plane_corners(:, 3), 'g');
end

h_legend = legend( ...
    [h_yCurr, h_yDes, h_JTruePlane, h_J1Plane, h_J2Plane], ...
    'yCurr', 'yDes', 'J True', 'J1', 'J2');

%% Get Suggested Actions
max_action_norm = 1;
desired_action = y_des - y_curr;
suggested_actions = zeros(2, num_arms);

J1_action = pinv(J1)*desired_action;
if (norm(J1_action) > max_action_norm)
    J1_action = J1_action * (max_action_norm / norm(J1_action));
end
suggested_actions(:, 1) = J1_action;

J2_action = pinv(J2)*desired_action;
if (norm(J2_action) > max_action_norm)
    J2_action = J2_action * (max_action_norm / norm(J2_action));
end
suggested_actions(:, 2) = J2_action;

%% Evaluate True Reward of each Arm
error_prev = norm(y_des - y_curr);
true_results    = zeros(3, num_arms);
true_errors     = zeros(num_arms, 1);
true_rewards    = zeros(num_arms, 1);

for arm_ind = 1:num_arms
    true_results(:, arm_ind) = J_true * suggested_actions(:, arm_ind);
    true_errors(arm_ind)  = norm(y_des - (y_curr + true_results(:, arm_ind)));
    true_rewards(arm_ind) = error_prev - true_errors(arm_ind);
end

%% Pick an arm
arm_to_pull = 1;
action = suggested_actions(:, arm_to_pull);
true_result = true_results(:, arm_to_pull);
true_reward = true_rewards(arm_to_pull);

predicted_results = zeros(3, num_arms);
predicted_results(:, 1) = J1 * action;
predicted_results(:, 2) = J2 * action;

%% Estimate KF inputs
J1_pred_result = predicted_results(:, 1);
h_J1Pred = line([y_curr(1), y_curr(1) + J1_pred_result(1)], [y_curr(2), y_curr(2) + J1_pred_result(2)], [y_curr(3), y_curr(3) + J1_pred_result(3)]);
set(h_J1Pred, 'Color', 'm', 'LineWidth', 4);

J2_pred_result = predicted_results(:, 2);
h_J2Pred = line([y_curr(1), y_curr(1) + J2_pred_result(1)], [y_curr(2), y_curr(2) + J2_pred_result(2)], [y_curr(3), y_curr(3) + J2_pred_result(3)]);
set(h_J2Pred, 'Color', 'g', 'LineWidth', 4);

h_JtrueResult = line([y_curr(1), y_curr(1) + true_result(1)], [y_curr(2), y_curr(2) + true_result(2)], [y_curr(3), y_curr(3) + true_result(3)]);
set(h_JtrueResult, 'Color', 'c', 'LineWidth', 4);

%%
% Estimated Reward
norm_true_movement_to_predicted = zeros(num_arms, 1);
norm_improvement                = zeros(num_arms, 1);
estimtated_improvement          = zeros(num_arms, 1);
estimated_rewards               = zeros(num_arms, 1);
true_improvement                = true_rewards - true_rewards(arm_to_pull);

for arm_ind = 1:num_arms
    norm_true_movement_to_predicted(arm_ind) = norm(true_result - predicted_results(:, arm_ind));
end

norm_to_arm_chosen = norm_true_movement_to_predicted(arm_to_pull);
for arm_ind = 1:num_arms
    norm_improvement(arm_ind) = norm_to_arm_chosen - norm_true_movement_to_predicted(arm_ind);
    estimtated_improvement(arm_ind) = norm_improvement(arm_ind) * abs(true_reward);
    estimated_rewards(arm_ind) = true_reward + estimtated_improvement(arm_ind);
end

fprintf('True rwrd | est. rwrd | nrm t vs p | true imp. | est imp.');
[true_rewards, estimated_rewards, norm_true_movement_to_predicted, true_improvement, estimtated_improvement]

%%
% Transition Noise
transition_noise = eye(2,2);
transition_noise(1,2) = J1_action' * J2_action / (norm(J1_action) * norm(J2_action));
transition_noise(2,1) = J1_action' * J2_action / (norm(J1_action) * norm(J2_action));

%%
fig = figure(1);
dist = 0:0.001:1;
plot(dist, 1 - exp(-dist));