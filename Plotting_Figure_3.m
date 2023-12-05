close all
clear all
clc

%% 1) Load data fromthe .mat Matlab data file or run the Load data script
% Option 1:
load('Pure_DM_Poros_GammaB_data.mat')
% 
%
% 
% 
% 
% Option 2:
% run('Load_Pure_DM_Poros_GammaB_data.m')
%
%

%% 2) Normalizing and unit conversion calculation

% Set all parameters
rhos = 3300;                 % density of solid
rhof = 2800;                 % density of fluid
deltarho = rhos -rhof;
Fmax = 0.2;                  % total amount of depletion
mu0 = 1.;                    % viscosity of fluid
g = 9.81;                    % gravitational acceleration [m/s2]
h = 100000;                  % 100km domain space [m]
eta0 = 1e19;                 % reference background solid viscosity [Pa s]
kappa = 7.272e-7;        % reference thermal diffusivity m2/s
cmy = 100*365.25*24*60*60;

% Reference Permeability array
K_0 = [4*10^(-7), 4*10^(-8), 4*10^(-9), 4*10^(-7), 4*10^(-8), 4*10^(-9), 4*10^(-7), 4*10^(-8), 4*10^(-9)];
% Half spreading rate array
U_0 = [2,2,2,4,4,4,8,8,8];

% Reference porosity
phi_0_U2K7 = ((rhos*U_0(1)*Fmax*mu0)/(rhof*K_0(1)*deltarho*g*cmy))^(1/3);%0.0042;
phi_0_U2K8 = ((rhos*U_0(2)*Fmax*mu0)/(rhof*K_0(2)*deltarho*g*cmy))^(1/3);%0.0091;
phi_0_U2K9 = ((rhos*U_0(3)*Fmax*mu0)/(rhof*K_0(3)*deltarho*g*cmy))^(1/3);%0.0197;
phi_0_U4K7 = ((rhos*U_0(4)*Fmax*mu0)/(rhof*K_0(4)*deltarho*g*cmy))^(1/3);%0.0053;
phi_0_U4K8 = ((rhos*U_0(5)*Fmax*mu0)/(rhof*K_0(5)*deltarho*g*cmy))^(1/3);%0.0115;
phi_0_U4K9 = ((rhos*U_0(6)*Fmax*mu0)/(rhof*K_0(6)*deltarho*g*cmy))^(1/3);%0.0248;
phi_0_U8K7 = ((rhos*U_0(7)*Fmax*mu0)/(rhof*K_0(7)*deltarho*g*cmy))^(1/3);%0.0067;
phi_0_U8K8 = ((rhos*U_0(8)*Fmax*mu0)/(rhof*K_0(8)*deltarho*g*cmy))^(1/3);%0.0145;
phi_0_U8K9 = ((rhos*U_0(9)*Fmax*mu0)/(rhof*K_0(9)*deltarho*g*cmy))^(1/3);%0.0312;

phi_0_vector = [phi_0_U2K7, phi_0_U2K8, phi_0_U2K9, phi_0_U4K7, phi_0_U4K8, phi_0_U4K9, ...
                phi_0_U8K7, phi_0_U8K8, phi_0_U8K9];
n = 3; % permeability-porosity exponent
m = 1; % compaction viscosity-porosity exponent
n_viscos_0 = 10^19; % reference background shear viscosity

w_0_vector = 33/28.*U_0./phi_0_vector*0.2; % reference fluid velocity
w_0_vector_m_Myr = w_0_vector * 10000;  % [cm/yr] -> m/Myr

% redimensionaling the freezing rate
GammaB_factor_vector = rhos .* phi_0_vector .* w_0_vector_m_Myr / h;
Porosity_factor_vector = phi_0_vector;

% Scaling
% Poros = Poros.*Porosity_factor_vector;
% GammaB = GammaB*GammaB_factor_vector

All_X = [U2K7_X, U2K8_X, U2K9_X,...
         U4K7_X, U4K8_X, U4K9_X,...
         U8K7_X, U8K8_X, U8K9_X];

All_Y = [U2K7_Y, U2K8_Y, U2K9_Y,...
         U4K7_Y, U4K8_Y, U4K9_Y,...
         U8K7_Y, U8K8_Y, U8K9_Y];

All_GammaB = [U2K7_GammaB, U2K8_GammaB, U2K9_GammaB,...
             U4K7_GammaB, U4K8_GammaB, U4K9_GammaB,...
             U8K7_GammaB, U8K8_GammaB, U8K9_GammaB];

All_Poros = [U2K7_Poros, U2K8_Poros, U2K9_Poros,...
             U4K7_Poros, U4K8_Poros, U4K9_Poros,...
             U8K7_Poros, U8K8_Poros, U8K9_Poros];

All_DM = [U2K7_DegreeM, U2K8_DegreeM, U2K9_DegreeM,...
             U4K7_DegreeM, U4K8_DegreeM, U4K9_DegreeM,...
             U8K7_DegreeM, U8K8_DegreeM, U8K9_DegreeM];

for ii = 1:9
    All_Poros(:,ii) = All_Poros(:,ii).*Porosity_factor_vector(ii);
    All_GammaB(:,ii) = All_GammaB(:,ii).*GammaB_factor_vector(ii);
end


% Put data in Array form
LAB_X_Array = zeros(100,9);
LAB_Y_Array = zeros(100,9);
LAB_GammaB_Array = zeros(100,9);
LAB_Poros_Array = zeros(100,9);
LAB_DegreeM_Array = zeros(100,9);
for k = 1:9
    [LAB_X, LAB_Y, Normalized_Y, LAB_GammaB_vector, LAB_Poros_vector, LAB_DegreeM_vector] = get_params(All_X(:,k), All_Y(:,k), All_GammaB(:,k), All_Poros(:,k), All_DM(:,k));
    LAB_X_Array(:,k) = LAB_X;
    LAB_Y_Array(:,k) = LAB_Y;
    LAB_GammaB_Array(:,k) = LAB_GammaB_vector;
    LAB_Poros_Array(:,k) = LAB_Poros_vector;
    LAB_DegreeM_Array(:,k) = LAB_DegreeM_vector;

end

% Sanity plot to see the LAB locations
% close all
% scatter(LAB_X, LAB_Y)


%% 3) Generating Plot here:

close all

% 1) Table of index k and the corresponding model
%        {k=1, 'U2K7'}, 
%        {k=2, 'U2K8'}, 
%        {k=3, 'U2K9'},
%        {k=4, 'U4K7'}, 
%        {k=5, 'U4K8'}, 
%        {k=6, 'U4K9'},
%        {k=7, 'U8K7'}, 
%        {k=8, 'U8K8'}, 
%        {k=9, 'U8K9'}


%
% 2) Get Figure 3 vertical profile

titles = {'U2K7', 'U2K8', 'U2K9', 'U4K7', 'U4K8', 'U4K9', 'U8K7', 'U8K8', 'U8K9'};

for k=[1,3,4,6,7,9]    % Figure 3 uses k = [1,3,4,6,7,9]
    
    plotting_Figure3(LAB_GammaB_Array(:,k), LAB_Poros_Array(:,k), LAB_DegreeM_Array(:,k), ...
                        Normalized_Y, 11000,0.22, titles{k})
end




%% Functions:

function [LAB_X, LAB_Y, Normalized_Y, LAB_GammaB_vector, LAB_Poros_vector, LAB_DegreeM_vector] = get_params(X, Y, GammaB, Poros, DegreeMelt)
    % %%Inputs%%
    % X         : the position in x direction
    % Y         : the position in y direction
    % GammaB    : Gammab data
    % Poros     : Porosity data
    % DegreeMelt: Degree Melt data

    LAB_X = zeros(100,1);
    LAB_Y = zeros(100,1);    
    LAB_GammaB_vector = zeros(100,1);
    LAB_Poros_vector = zeros(100,1);
    LAB_DegreeM_vector = zeros(100,1);

    position_ind = 1;
    step = 0.005;

    Normalized_Y = -0.499:step:0;

    for i = -0.499:step:-step
        in_range_value_mask = Y > (i) & Y < (i+step) & X >= 0;

        % GammaB
        in_range_GammaB = GammaB(in_range_value_mask);
        [value, gamma_ind] = min(in_range_GammaB);
        LAB_GammaB_vector(position_ind) = (-1)*value;

        % Poros
        in_range_Poros = Poros(in_range_value_mask);
        [value, poros_ind] = max(in_range_Poros);
        LAB_Poros_vector(position_ind) = value;

        % DegreeMelt
        in_range_DegreeM = DegreeMelt(in_range_value_mask & (X>=0.9));
        LAB_DegreeM_vector(position_ind) = (-1)*mean(in_range_DegreeM);

        % Y
        in_range_Y = Y(in_range_value_mask);
        LAB_Y(position_ind) = in_range_Y(poros_ind);

        % X
        in_range_X = X(in_range_value_mask);
        LAB_X(position_ind) = in_range_X(poros_ind);

        position_ind = position_ind+1;
    end
end



function plotting_Figure3(GammaB, Poros, DM, Normalized_Y, GammaB_Xlim, Poros_Xlim, title_name)
    % %%Inputs%%
    % GammaB       : Normalized Gammab data for plotting
    % Poros        : Normalized Porosity data
    % DM           : Normalized Degree Melt data
    % Normalized_Y : Y positions
    % GammaB_Xlim  : Max x-limit for your plot for the value Gamma
    % Poros_Xlim   : Max x-limit for your plot for the value Poros
    % title_name   : Title of the plot in text

    iii = Normalized_Y;
    figure('Position', [100, 100, 400, 700])
    set(gcf,'color','w');
    
    
    % Set up main axis (on top)
    clf()
    t = tiledlayout(1, 2);
    
    p1 = axes(t);
    l1 = plot(p1, abs(GammaB), iii, 'Color', "#EDB120", 'LineWidth', 1.4,'DisplayName','GammaB');
    p1.XLim = [0,GammaB_Xlim];
    
    p2 = axes(t);
    l2 = plot(p2, abs(Poros), iii,'Color', "#D95319", 'DisplayName','Porosity','LineWidth', 1.4);
    p2.XLim = [0,Poros_Xlim];
    p2.Color = 'none';
    
    p3 = axes(t);
    l3 = plot(p3, DM, iii, 'Color', "#0072BD", 'LineStyle',"--", 'DisplayName','DegreeM','LineWidth', 1.4);
    p3.XLim = [-0.25,0];
    p3.Color = 'none';
    
    p1. XColor = 'white';
    p2. XColor = 'white';
    p3. XColor = 'white';
    p2. YColor = 'white';
    p3. YColor = 'white';
    p1.Box = 'off';
    p2.Box = 'off';
    p3.Box = 'off';
    names = legend([l1;l2;l3] ,'Location','bestoutside', 'Box', 'off');
    set(names, 'Position', [0.48,0.03,0.25,0.07]);
    title(title_name);
    % 
    ax = axes(t);
    ax.Color = 'none';
    ax.XTick = 0:2750:11000; 
    ax.YLim = [-0.5,0]; 
    ax.XLim = [0,11000]; 

    % Set axis limits for all other axes
    additionalAxisLimits = [...
        0, 0.22;       % axis 2 for Porosity
        -0.25, 0];     % axis 3 for Degree Melt
    
    % Compute tick values for each axis to align with main axis
    additionalAxisTicks = arrayfun(@(i){linspace(additionalAxisLimits(i,1), ...
        additionalAxisLimits(i,2),numel(ax.XTick))}, 1:size(additionalAxisLimits,1));
    
    % Set up multi-line ticks
    allTicks = [ax.XTick; cell2mat(additionalAxisTicks')];
    tickLabels = compose('%4d\\newline%.2f\\newline%.2f', allTicks(:).'); 
    
    ax.Position(3:4) = ax.Position(3:4) * .75; % Reduced to 75%
    ax.Position(2) = ax.Position(2) + .2;
    
    
    
    % Add x tick labels
    set(ax, 'XTickLabel', tickLabels, 'TickDir', 'out', 'XTickLabelRotation', 0);
end




