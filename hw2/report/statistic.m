%% CAR

filename = 'hw2_SRCC.txt';
f = fopen(filename, 'r');

cap = [];
carTime = [];
waitTime = [];

for i = 1:50,
    str = fgets(f);
    element = strsplit(str, ' ');
    C = cellstr(element);
    cap = [cap str2double(C(1))];
    carTime = [carTime str2double(C(2))];
    waitTime = [waitTime str2double(C(3))];
end

cap
carTime
waitTime

select = [];
for i = 1:4,
    carBegin = i;
    select = [select; waitTime(carBegin:5:50)];
end

bar(select);
set(gca,...
    'XTickLabel',[1 10 100 1000],...
    'XTick', 1:4);
legend('1', '2', '3', '4', '5', '6', '7', '8', '9', '10');


xlabel('Car running time(sec)', 'FontSize', 16); 
ylabel('Average waiting time (sec)', 'FontSize', 16);

%% threads
clear;
filename = ['first/hw2_NB_BHalgo.txt first/hw2_NB_BHalgo_1.txt first/hw2_NB_openmp.txt first/hw2_NB_pthread.txt'];
C = strsplit(filename, ' ');
name = cellstr(C);
f(1) = fopen('first/hw2_NB_BHalgo.txt', 'r');
f(2) = fopen('first/hw2_NB_BHalgo_1.txt', 'r');
f(3) = fopen('first/hw2_NB_openmp.txt', 'r');
f(4) = fopen('first/hw2_NB_pthread.txt', 'r');

threads = [];
times = [];
total = [];
for i = 1:4,
    times = [];
    for j = 1:40,
        str = fgets(f(i));
        element = strsplit(str, ' ');
        CC = cellstr(element);
        times = [times str2double(CC(3))]; 
    end
    total = [total; times];
end

plot(1:40, total)
ylabel('run time(sec)', 'FontSize', 16); 
xlabel('num of threads', 'FontSize', 16);
legend('BHalgo theta = 0.1', 'BHalgo theta = 1', 'openmp', 'pthread');

%% steps
clear;
f(1) = fopen('second/hw2_NB_BHalgo.txt', 'r');
f(2) = fopen('second/hw2_NB_BHalgo_1.txt', 'r');
f(3) = fopen('second/hw2_NB_openmp.txt', 'r');
f(4) = fopen('second/hw2_NB_pthread.txt', 'r');

steps = [];
times = [];
total = [];
for i = 1:4,
    times = [];
    for j = 1:8,
        str = fgets(f(i))
        element = strsplit(str, ' ');
        CC = cellstr(element);
        steps = [steps str2double(CC(3))];
        times = [times str2double(CC(3))]; 
    end
    total = [total; times];
end

plot(1:8, total)
xlabel('steps(x100)', 'FontSize', 16); 
ylabel('runtime (sec)', 'FontSize', 16);
legend('BHalgo theta = 0.1', 'BHalgo theta = 1', 'openmp', 'pthread');\


%% N
clear;
f(1) = fopen('third/hw2_NB_BHalgo.txt', 'r');
f(2) = fopen('third/hw2_NB_BHalgo_1.txt', 'r');
f(3) = fopen('third/hw2_NB_openmp.txt', 'r');
f(4) = fopen('third/hw2_NB_pthread.txt', 'r');

steps = [];
times = [];
total = [];
for i = 1:4,
    times = [];
    for j = 1:4,
        str = fgets(f(i))
        element = strsplit(str, ' ');
        CC = cellstr(element);
        % steps = [steps str2double(CC(3))];
        times = [times str2double(CC(3))]; 
    end
    total = [total; times];
end

plot( total')
xlabel('N(input size)', 'FontSize', 16); 
ylabel('runtime (sec)', 'FontSize', 16);
legend('BHalgo theta = 0.1', 'BHalgo theta = 1', 'openmp', 'pthread');
set(gca,...
    'XTickLabel',[2 945 5172 6117],...
    'XTick', 1:4);

%% theta

clear;
f(1) = fopen('hw2_NB_BHalgo_grid.txt', 'r');

total = [];
io = [];
theta = [];
build = [];
compute = []
for i = 1:1,
    times = [];
    for j = 1:10,
        str = fgets(f(i))
        element = strsplit(str, ' ');
        CC = cellstr(element);
        theta = [theta str2double(CC(1))];
        io = [io str2double(CC(2))];
        build = [build str2double(CC(3))];
        compute = [compute str2double(CC(4))];
    end
    total = [total; times];
end



bar([io;build;compute]', 'stack');
xlabel('theta', 'FontSize', 16); 
ylabel('runtime (sec)', 'FontSize', 16);
legend('IO', 'Build Tree', 'Compute');
set(gca,...
    'XTickLabel',[0.1:0.1:1.0],...
    'XTick', 1:10);
colormap('autumn');
