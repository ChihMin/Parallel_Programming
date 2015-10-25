clear;

[x_1, y_1] = statistic('advanced_report.txt', 1, 48, 0);
[x_2, y_2] = statistic('advanced_report.txt', 1, 96, 48);
[x_3, y_3] = statistic('advanced_report.txt', 1, 144, 96);
[x_4, y_4] = statistic('advanced_report.txt', 1, 192, 144);

plot(x_1, y_1, 'b-o', x_2, y_2, 'r-o', x_3, y_3, 'k-o', x_4, y_4, 'm-o');
axis([0,49,0, 30]);
 set(gca,...
    'XTickLabel',1:48,...
    'XTick', 1:48);
legend('N=100000', 'N=1000000', 'N=10000000', 'N=84000000');
title('Advanced Version', 'FontSize', 16);
xlabel('Process number', 'FontSize', 16); 
ylabel('Execution time(s)', 'FontSize', 16);
   
