cuda = load('report/HW4_102062111_cuda.log');
openmp = load('report/HW4_102062111_openmp.log');
mpi = load('report/HW4_102062111_mpi.log');

total(1,:) = cuda(:, 1) / 1000;
total(2,:) = openmp(:, 1) / 1000;
total(3,:) = mpi(:, 1) / 1000;

plot(total');
title('total time');
legend('cuda', 'openmp', 'mpi');
xlabel('testcase Size(N)', 'FontSize', 16); 
ylabel('run time(seconds)', 'FontSize', 16);

set(gca,...
     'XTickLabel',[1024:1024:9216],...
     'XTick', [1:9]);

%%
figure;
compute(1,:) = cuda(:, 2) / 1000;
compute(2,:) = openmp(:, 2) / 1000;
compute(3,:) = mpi(:, 2) / 1000;

plot(compute');
title('computing time');
legend('cuda', 'openmp', 'mpi');
xlabel('testcase Size(N)', 'FontSize', 16); 
ylabel('run time(seconds)', 'FontSize', 16);

set(gca,...
     'XTickLabel',[1024:1024:9216],...
     'XTick', [1:9]);
 
 %%
 
 figure;
io(1,:) = cuda(:, 3) / 1000;
io(2,:) = openmp(:, 3) / 1000;
io(3,:) = mpi(:, 4) / 1000;

plot(io');
title('memory time');
legend('cuda', 'openmp', 'mpi');
xlabel('testcase Size(N)', 'FontSize', 16); 
ylabel('run time(seconds)', 'FontSize', 16);

set(gca,...
     'XTickLabel',[1024:1024:9216],...
     'XTick', [1:9]);
 
 %%
figure;
io(1,:) = cuda(:, 4) / 1000;
io(2,:) = openmp(:, 4) / 1000;
io(3,:) = mpi(:, 5) / 1000;

plot(io');
title('IO time');
legend('cuda', 'openmp', 'mpi');
xlabel('testcase Size(N)', 'FontSize', 16); 
ylabel('run time(seconds)', 'FontSize', 16);

set(gca,...
     'XTickLabel',[1024:1024:9216],...
     'XTick', [1:9]);
 
 %% Communication time 
 figure;

comm(1,:) = mpi(:, 3) / 1000;

plot(comm);
title('Communication time');
legend('mpi');
xlabel('testcase Size(N)', 'FontSize', 16); 
ylabel('run time(seconds)', 'FontSize', 16);

set(gca,...
     'XTickLabel',[1024:1024:9216],...
     'XTick', [1:9]);
 %%
 
cuda_fast = load('report/HW4_102062111_fast_cuda.log');
openmp_fast = load('report/HW4_102062111_fast_openmp.log');
mpi_fast = load('report/HW4_102062111_fast_mpi.log');

total(1,:) = cuda_fast(:, 2) / 1000;
total(2,:) = openmp_fast(:, 2) / 1000;
total(3,:) = cuda(:, 2) / 1000;
total(4,:) = openmp(:, 2) / 1000;

plot(total');
title('computing time');
legend('cuda\_optimize', 'openmp\_optimize', 'cuda', 'openmp');
xlabel('testcase Size(N)', 'FontSize', 16); 
ylabel('run time(seconds)', 'FontSize', 16);

set(gca,...
     'XTickLabel',[1024:1024:9216],...
     'XTick', [1:9]);
%%

cuda_kernel = load('report/HW4_102062111_cuda_block_kernel.log');
openmp_kernel = load('report/HW4_102062111_openmp_block_kernel.log');
mpi_kernel = load('report/HW4_102062111_mpi_block_kernel.log');


N = 1024
cuda_gflops = [];
openmp_gflops = [];
mpi_gflops = [];
for i = 1:6,
    blockSize = 2 ^ (i + 4);
    blockNum = N / blockSize;
    cuda_gflops(i) = 3 * N ^ 3 / 10^9 / (cuda_kernel(i,2) / 10^3);
    openmp_gflops(i) = 3 * N ^ 3 / 10^9 / (openmp_kernel(i,2) / 10^3);
    mpi_gflops(i) = 3 * N ^ 3 / 10^9 / (mpi_kernel(i,2) / 10^3);
end

total = [cuda_gflops; openmp_gflops; mpi_gflops]';

label = cuda_block(:,1);
bar(total);
title('GFLOPS');
legend('cuda', 'openmp', 'mpi');
xlabel('Block Factor', 'FontSize', 16); 
ylabel('GFLOPS', 'FontSize', 16);

set(gca,...
     'XTickLabel',label,...
     'XTick', 1:6); 
 
 %%

cuda_kernel = load('report/HW4_102062111_cuda_block_memory.log');
openmp_kernel = load('report/HW4_102062111_openmp_block_memory.log');
mpi_kernel = load('report/HW4_102062111_mpi_block_memory.log');


N = 1024
cuda_gflops = [];
openmp_gflops = [];
mpi_gflops = [];
for i = 1:6,
    blockSize = 2 ^ (i + 4);
    blockNum = N / blockSize;
    
    % cuda_gflops(i) = blockNum * N ^ 2 / (cuda_kernel(i,2) / 10^3) / 10^9;
    openmp_gflops(i) = blockNum * N ^ 2 / (openmp_kernel(i,2) / 10^3) / 10^9;
    mpi_gflops(i) = blockNum * N ^ 2 / (mpi_kernel(i,2) / 10^3) / 10^9;
end

total = [openmp_gflops; mpi_gflops]';

label = cuda_block(:,1);
plot(total);
title('Memory BandWidth');
legend('openmp', 'mpi');
xlabel('Block Factor', 'FontSize', 16); 
ylabel('BandWidth(GB/s)', 'FontSize', 16);

set(gca,...
     'XTickLabel',label,...
     'XTick', 1:6); 