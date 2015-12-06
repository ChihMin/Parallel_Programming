clear;

[x_1, y_1] = statistic('advanced_report.txt', 1, 96, 48, 12);
[x_2, y_2] = statistic('advanced_report.txt', 1, 96, 60, 12);
[x_3, y_3] = statistic('advanced_report.txt', 1, 96, 72, 12);
[x_4, y_4] = statistic('advanced_report.txt', 1, 96, 84, 12);

subplot(2,2,1);
Y = [y_1'];
hg = bar(Y);
set(gca,...
    'XTickLabel',x_1,...
    'XTick', 1:12);
set(hg(1), 'FaceColor','r');
legend('basic without judge', 'basic ');


title('Basic, N = 1000000, nodes = 1', 'FontSize', 16);
xlabel('Process per node', 'FontSize', 16); 
ylabel('Execution time(s)', 'FontSize', 16);

subplot(2,2,2);
Y = [y_2'];
hg = bar(Y);
set(gca,...
    'XTickLabel',x_1,...
    'XTick', 1:12);
set(hg(1), 'FaceColor','r');
legend('basic without judge', 'basic ');


title('Basic, N = 1000000, nodes = 2', 'FontSize', 16);
xlabel('Process per node', 'FontSize', 16); 
ylabel('Execution time(s)', 'FontSize', 16);

subplot(2,2,3);
Y = [y_3'];
hg = bar(Y);
set(gca,...
    'XTickLabel',x_1,...
    'XTick', 1:12);
set(hg(1), 'FaceColor','r');
legend('basic without judge', 'basic ');


title('Basic, N = 1000000, nodes = 3', 'FontSize', 16);
xlabel('Process per node', 'FontSize', 16); 
ylabel('Execution time(s)', 'FontSize', 16);

subplot(2,2,4);
Y = [y_4'];
hg = bar(Y);
set(gca,...
    'XTickLabel',x_1,...
    'XTick', 1:12);
set(hg(1), 'FaceColor','r');
legend('basic without judge', 'basic ');


title('Basic, N = 1000000, nodes = 4', 'FontSize', 16);
xlabel('Process per node', 'FontSize', 16); 
ylabel('Execution time(s)', 'FontSize', 16);