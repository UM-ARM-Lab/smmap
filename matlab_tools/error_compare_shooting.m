clc; clear;
%%
%%
experiment = 'colab_folding';
base_dir = ['../logs/' experiment '/'];

%%
shooting_error = load( [base_dir 'shooting_method/error.txt'] );
time = load( [base_dir 'shooting_method/time.txt'] );
time = time - time(1);

pure_jacobian_error = load( [base_dir 'deformability_range_10_to_20/trans_14_rot_14/error.txt'] );

%%
% https://dgleich.github.io/hq-matlab-figs/
% http://blogs.mathworks.com/loren/2007/12/11/making-pretty-graphs/

width = 7;      % Width in inches
height = 6;     % Height in inches
alw = 0.7;      % AxesLineWidth
fsz = 18;       % Fontsize
lw = 1;         % LineWidth
msz = 12;       % MarkerSize

%%
close all;
fig = figure( 'Units', 'inches', ...
              'Position', [0, 0, width, height] );
set( fig, 'PaperPositionMode', 'auto' );

plot( time, [ shooting_error, pure_jacobian_error ] )
h_legend = legend( ['Pure Jacobian' repmat(char(3), 1, 1)], 'Shooting' );

h_Xlabel = xlabel( 'Time (s)' );
h_Ylabel = ylabel( 'Error' );

set([h_Xlabel, h_Ylabel, h_legend], ...
    'FontName'   , 'Helvetica');
set([gca, h_legend]            , ...
    'FontSize'   , fsz-2       );
set([h_Xlabel, h_Ylabel]       , ...
    'FontSize'   , fsz         );

output_name = ['output_images/' experiment '/shooting-comparison.eps' ];
print( output_name, '-depsc2', '-r300');