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



