function [ x, y ] = statistic(filename, range_a, range_b, thred, len)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    f = fopen(filename, 'r');

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
%{
    process(1:50) = 1;
    num_of_case(1:50) = 0;
    for i = 1:len,
        target = nodes(i) * ppn(i);
        process(target) = process(target) * time(i);
        num_of_case(target) = num_of_case(target) + 1;
    end

    x_advance = [];
    y_advance = [];
    for i = 1:len,
        if num_of_case(i) > 0,
            x_advance = [x_advance i];
            process(i) = process(i) .^ (1/num_of_case(i));
            y_advance = [y_advance process(i)];
            fprintf('%d %f\n', i, process(i));
        end
    end
%}
    fclose(f);
    x = ppn(1:len);
    y = time(1:len);
end

