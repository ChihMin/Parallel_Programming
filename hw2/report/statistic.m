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


xlabel('Capacity', 'FontSize', 16); 
ylabel('Average waiting time (sec)', 'FontSize', 16);