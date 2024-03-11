x = readcell('p2p.xlsx','Sheet','plot','Range','A2:A1002');
U2K7 = readcell('p2p.xlsx','Sheet','plot','Range','B2:B1002');
U2K8 = readcell('p2p.xlsx','Sheet','plot','Range','C2:C1002');
U2K9 = readcell('p2p.xlsx','Sheet','plot','Range','D2:D1002');

U4K7 = readcell('p2p.xlsx','Sheet','plot','Range','E2:E1002');
U4K8 = readcell('p2p.xlsx','Sheet','plot','Range','F2:F1002');
U4K9 = readcell('p2p.xlsx','Sheet','plot','Range','G2:G1002');

U6K7 = readcell('p2p.xlsx','Sheet','plot','Range','H2:H1002');
U6K8 = readcell('p2p.xlsx','Sheet','plot','Range','I2:I1002');
U6K9 = readcell('p2p.xlsx','Sheet','plot','Range','J2:J1002');

U8K7 = readcell('p2p.xlsx','Sheet','plot','Range','K2:K1002');
U8K8 = readcell('p2p.xlsx','Sheet','plot','Range','L2:L1002');
U8K9 = readcell('p2p.xlsx','Sheet','plot','Range','M2:M1002');

%% Figure 2
clc
close all
figure('Position', [150, 0, 550, 850])
set(gcf,'color','w');
set(gcf,'DefaultLineLineWidth',2);
hold on
plot(cell2mat(U2K7),cell2mat(x),'-', 'Color','#F05039')
% plot(cell2mat(U2K8),cell2mat(x),'b.-')
% plot(cell2mat(U2K8),cell2mat(x),'b:')
plot(cell2mat(U2K9),cell2mat(x),'--',  'Color','#F05039')
plot(cell2mat(U4K7),cell2mat(x),'-',  'Color','#1F449C')
% plot(cell2mat(U4K8),cell2mat(x),'g.-')
% plot(cell2mat(U4K8),cell2mat(x),'g:')
plot(cell2mat(U4K9),cell2mat(x),'--',  'Color','#1F449C')
% plot(cell2mat(U6K7),cell2mat(x),'r-')
% plot(cell2mat(U6K8),cell2mat(x),'r.-')
% plot(cell2mat(U6K8),cell2mat(x),'r:')
% plot(cell2mat(U6K9),cell2mat(x),'r--')
plot(cell2mat(U8K7),cell2mat(x),'-', 'Color','#A8B6CC')
% plot(cell2mat(U8K8),cell2mat(x),'k.-')
% plot(cell2mat(U8K8),cell2mat(x),'k:')
plot(cell2mat(U8K9),cell2mat(x),'--', 'Color','#A8B6CC')
xlabel('Degree Melt')
ylabel('Height')
legend({'U2K7','U2K9','U4K7','U4K9','U8K7','U8K9'},'Location','northeastoutside','Box','off')
ylim([0.5,1])
xlim([-0.22, 0])






