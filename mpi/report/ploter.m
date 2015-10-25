clear;

[x_1, y_1] = statistic('basic_report.txt', 1, 96, 48);
[x_2, y_2] = statistic('basic_bcast_report.txt', 1, 84, 36);

plot(x_1, y_1, 'b-o', x_2, y_2, 'r-o');
% axis([0,49,0, 50]);
 set(gca,...
    'XTickLabel',1:48,...
    'XTick', 1:48);
legend('basic without judge', 'basic ');
title('Basic Broadcase overhead experienment, N = 100000', 'FontSize', 16);
xlabel('Process number', 'FontSize', 16); 
ylabel('Execution time(s)', 'FontSize', 16);
   
figure;
Y = [y_1'  y_2'];
hg = bar(Y);
set(gca,...
    'XTickLabel',x_1,...
    'XTick', 1:29);
set(hg(2), 'FaceColor','r');
legend('basic without judge', 'basic ');


title('Basic Broadcase overhead experienment, N = 100000', 'FontSize', 16);
xlabel('Process number', 'FontSize', 16); 
ylabel('Execution time(s)', 'FontSize', 16);