%% strong

clear;
filename = ['MS_MPI_static MS_MPI_dynamic MS_OpenMP_static MS_OpenMP_dynamic'];
filename = strsplit(filename, ' ');

data(1:4, 1:40) = 0;
for i = 1:4, 
    str = ['strong/' char(filename(i)) '.txt'];
    f = fopen(str);
    
    while ~feof(f)
        ret = fscanf(f, '%f %f', 2); 
        if ~isempty(ret),
            thread = ret(1);
            times = ret(2);
            data(i, thread) = times;
        end
    end
    
    % A = [A fscanf(f, '%f %f')];  
    fclose(f);
end

plot(data');
set(gca,...
    'XTick', 1:40);
legend('MPI Static', 'MPI Dynamic', 'OpenMP Static', 'OpenMP Dynamic');
xlabel('Number of threads/ranks', 'FontSize', 16); 
ylabel('time(sec)', 'FontSize', 16);

%% Weak
clear;
filename = ['MS_MPI_static MS_MPI_dynamic MS_OpenMP_static MS_OpenMP_dynamic MS_Hybrid_static MS_Hybrid_dynamic'];
filename = strsplit(filename, ' ');
times(1:6, 1:15) = 0;
test(1:15) = 0;
for i = 1:6, 
    str = ['weak_N/' char(filename(i)) '.txt']
    f = fopen(str);
    
    while ~feof(f)
        ret = fscanf(f, '%d %f', 2); 
        if ~isempty(ret),
            N = ret(1);
            test(N/200) = N;
            time = ret(2);
            times(i, N/200) = time;
        end
    end
    
    % A = [A fscanf(f, '%f %f')];  
    fclose(f);
end

plot(times');
set(gca,...
    'XTickLabel', test,...
    'XTick', 1:15);
legend('MPI Static', 'MPI Dynamic', 'OpenMP Static', 'OpenMP Dynamic', 'Hybrid Static', 'Hybrid Dynamic');
xlabel('Number of Testcase(N)', 'FontSize', 16); 
ylabel('time(sec)', 'FontSize', 16);

%% 
clear;
filename = ['MS_MPI_static MS_MPI_dynamic MS_OpenMP_static MS_OpenMP_dynamic MS_Hybrid_static MS_Hybrid_dynamic'];
filename = strsplit(filename, ' ');
times(1:6, 1:12) = 0;
points(1:6, 1:12) = 0;
rank(1:6, 1:12) = 0;
for i = 1:6, 
    str = ['balance/' char(filename(i)) '.txt']
    f = fopen(str);
    
    while ~feof(f)
        str = fgets(f);
        element = strsplit(str, ' ');
        C = cellstr(element);
        j = str2double(C(3)) + 1;
        
        rank(i, j) = str2double(C(3));
        times(i, j) = str2double(C(5));
        points(i, j) = str2double(C(7));
        if i == 1,
            points(i, j) = points(i, j) * 5000;
        end
    end
    
    fclose(f);
end

bar(points');
title('Points');
set(gca,...
    'XTickLabel', rank(1,:),...
    'XTick', 1:12);
legend('MPI Static', 'MPI Dynamic', 'OpenMP Static', 'OpenMP Dynamic', 'Hybrid Static', 'Hybrid Dynamic');
xlabel('Thread/Rank Number', 'FontSize', 16); 
ylabel('Points', 'FontSize', 16);

%% Best
clear;
filename = 'MS_Hybrid_dynamic_10000';
filename = strsplit(filename, ' ');
data(1:4, 1:12) = double(1);
count(1:4, 1:12) = 0;
for i = 1:1,
    str = ['best/' char(filename(i)) '.txt']
    f = fopen(str);
    
    while ~feof(f)
        ret = fscanf(f, '%d %d %d %f', 4); 
        if ~isempty(ret),
            node = ret(1)
            ppn = ret(2)
            thread = ppn / ret(3)
            time = ret(4)
            data(node, ppn) = data(node, ppn) * time;
            count(node, ppn) = count(node, ppn) + 1;
        end
    end
    for i = 1:4,
        for j = 1:12,
            data(i, j) = data(i, j) .^ (1/count(i, j));
        end
    end
    
    fclose(f);
end
bar3(data');
set(gca,...
    'XTick', 1:4);
set(gca,...
    'YTick', 1:12);
xlabel('Node', 'FontSize', 16);
ylabel('PPN', 'FontSize', 16);
zlabel('Time(sec)', 'FontSize', 16);
colormap autumn;

%{
plot(times');
set(gca,...
    'XTickLabel', test,...
    'XTick', 1:15);
legend('MPI Static', 'MPI Dynamic', 'OpenMP Static', 'OpenMP Dynamic', 'Hybrid Static', 'Hybrid Dynamic');
xlabel('Number of Testcase(N)', 'FontSize', 16); 
ylabel('time(sec)', 'FontSize', 16);
%}