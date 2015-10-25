%%
range_a = 1;
range_b = 96;
thred = 48;

f = fopen('basic_report.txt', 'r');


nodes = [];
ppn = [];
testcase = [];
time = [];

for i = range_a:range_b,
    str = fgets(f);
    if i > thred,
        element = strsplit(str, ' ');
        C = cellstr(element);
        nodes = [nodes str2double(C(1))];
        ppn = [ppn str2double(C(2))];
        testcase = [testcase str2double(C(3))];
        time = [time str2double(C(4))];
    end
end

process(1:50) = 1;
num_of_case(1:50) = 0;
for i = 1:48,
    target = nodes(i) * ppn(i);
    process(target) = process(target) * time(i);
    num_of_case(target) = num_of_case(target) + 1;
end

x_basic = [];
y_basic = [];
for i = 1:48,
    if num_of_case(i) > 0,
        x_basic = [x_basic i];
        process(i) = process(i) .^ (1/num_of_case(i));
        y_basic = [y_basic process(i)];
        fprintf('%d %f\n', i, process(i));
    end
end

fclose(f);
%%

f = fopen('advanced_report.txt', 'r');

nodes = [];
ppn = [];
testcase = [];
time = [];

for i = range_a:range_b,
    if i > thred,
        str = fgets(f);
        element = strsplit(str, ' ');
        C = cellstr(element);
        nodes = [nodes str2double(C(1))];
        ppn = [ppn str2double(C(2))];
        testcase = [testcase str2double(C(3))];
        time = [time str2double(C(4))];
    end
end

process(1:50) = 1;
num_of_case(1:50) = 0;
for i = 1:48,
    target = nodes(i) * ppn(i);
    process(target) = process(target) * time(i);
    num_of_case(target) = num_of_case(target) + 1;
end

x_advance = [];
y_advance = [];
for i = 1:48,
    if num_of_case(i) > 0,
        x_advance = [x_advance i];
        process(i) = process(i) .^ (1/num_of_case(i));
        y_advance = [y_advance process(i)];
        fprintf('%d %f\n', i, process(i));
    end
end

fclose(f);

%%

plot(x_basic, y_basic, 'b-o', x_advance, y_advance, 'r-o');
axis([0,49,0, 1200]);
 set(gca,...
    'XTickLabel',1:48,...
    'XTick', 1:48);
legend('Basic', 'Advance');
title('N = 1000000', 'FontSize', 16);
xlabel('Process number', 'FontSize', 16); 
ylabel('Execution time(s)', 'FontSize', 16);
    
