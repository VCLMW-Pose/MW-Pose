%     Created on Fri Oct 5 17:00 2019
% 
%     Author           : Yu Du
%     Email            : yuduseu@gmail.com
%     Last edit date   :  2019
% 
% South East University Automation College
% Vision Cognition Laboratory, 211189 Nanjing China
filename = fullfile('logger_val.txt');
fid = fopen(filename);

C = textscan(fid,...
    '%n%n%n%n%n%n%n%n%n%n%n%n%n%n%n%n%n%n%n','HeaderLines',1);

plot(C{1},'r');
hold on;
plot(C{2},'g');
hold on;
plot(C{3},'b');
hold on;
plot(C{4},'Color',[1 0.56 0]);
hold on;
plot(C{5},'m');
hold on;
plot(C{6},'c');
hold on;
plot(C{7},'Color',[0.58 0.09 0.32]);
hold on;
plot(C{8},'Color',[0.48 0.51 1]);
hold off;
xlabel('Epoch');
ylabel('Loss');
ylim([0 0.005]);
legend('nose', 'neck', 'rshoulder','relbow','rwrist','lshould',...
    'lelbow','lwrist','Location', 'northeastoutside')
savefig('curve-I');

grid;
plot(C{9},'r');
hold on;
plot(C{10},'g');
hold on;
plot(C{11},'b');
hold on;
plot(C{12},'Color',[1 0.56 0]);
hold on;
plot(C{13},'m');
hold on;
plot(C{14},'c');
hold on;
plot(C{15},'Color',[0.58 0.09 0.32]);
hold on;
plot(C{16},'Color',[0.48 0.51 1]);
hold off;
xlabel('Epoch');
ylabel('Loss');
legend('rhip','rknee','rankle','lhip','lknee',...
    'lankle','reye','leye','Location', 'northeastoutside')
savefig('curve-II');

grid;
plot(C{17},'r');
hold on;
plot(C{18},'g');
hold off;
xlabel('Epoch');
ylabel('Loss');
legend('rear','lear','Location', 'northeastoutside')
savefig('curve-III');

grid;
plot(C{19},'m');
hold off;
xlabel('Epoch');
ylabel('Loss');
legend('total','Location', 'northeastoutside')
savefig('curve-total');


