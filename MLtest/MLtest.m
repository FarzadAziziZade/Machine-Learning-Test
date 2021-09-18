close all
clear all
clc
a=[]; inputs=[]; targetfourthroot=[]; x=[]; t=[]; y=[];
a=csvread('datain.csv');
% set mods -----------------------------
loopmode=false; % False= Run the code only once % True= Run the code multiple times
nubmerOfOutputs=2; % 1,2,3
numberOfHiddenLayers=1; % 1 or 2
%---------------------------------------
if nubmerOfOutputs==3
inputs=a(:,1:size(a,2)-nubmerOfOutputs)';
targetfourthroot=a(:,size(a,2)-nubmerOfOutputs+1:size(a,2))';
elseif nubmerOfOutputs==2
inputs=a(:,1:size(a,2)-nubmerOfOutputs-1)';
targetfourthroot=a(:,size(a,2)-nubmerOfOutputs+1-1:size(a,2)-1)';
elseif nubmerOfOutputs==1
inputs=a(:,1:size(a,2)-nubmerOfOutputs-2)';
targetfourthroot=a(:,size(a,2))';    
end
x = inputs;
t = targetfourthroot;
% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
%---------------------------------------------------------------
%   Neural Network Training Functions.
%  
%   To change a neural network's training algorithm set the net.trainFcn
%   property to the name of the corresponding function.  For example, to use
%   the scaled conjugate gradient backprop training algorithm:
%  
%     net.trainFcn = 'trainscg';
%  
%   Backpropagation training functions that use Jacobian derivatives
%  
%     These algorithms can be faster but require more memory than gradient
%     backpropation.  They are also not supported on GPU hardware.
%  
%     trainlm   - Levenberg-Marquardt backpropagation.
%     trainbr   - Bayesian Regulation backpropagation.
%  
%   Backpropagation training functions that use gradient derivatives
%  
%     These algorithms may not be as fast as Jacobian backpropagation.
%     They are supported on GPU hardware with the Parallel Computing Toolbox.
%  
%     trainbfg  - BFGS quasi-Newton backpropagation.
%     traincgb  - Conjugate gradient backpropagation with Powell-Beale restarts.
%     traincgf  - Conjugate gradient backpropagation with Fletcher-Reeves updates.
%     traincgp  - Conjugate gradient backpropagation with Polak-Ribiere updates.
%     traingd   - Gradient descent backpropagation.
%     traingda  - Gradient descent with adaptive lr backpropagation.
%     traingdm  - Gradient descent with momentum.
%     traingdx  - Gradient descent w/momentum & adaptive lr backpropagation.
%     trainoss  - One step secant backpropagation.
%     trainrp   - RPROP backpropagation.
%     trainscg  - Scaled conjugate gradient backpropagation.
%  
%   Supervised weight/bias training functions
%  
%     trainb    - Batch training with weight & bias learning rules.
%     trainc    - Cyclical order weight/bias training.
%     trainr    - Random order weight/bias training.
%     trains    - Sequential order weight/bias training.
%  
%   Unsupervised weight/bias training functions
%  
%     trainbu   - Unsupervised batch training with weight & bias learning rules.
%     trainru   - Unsupervised random order weight/bias training.
%---------------------------------------------------------------
% Create a Fitting Network
if numberOfHiddenLayers==2
    trainFcn = 'trainlm';
    hiddenLayer1Size = 4;
    hiddenLayer2Size = 8;
    net = fitnet([hiddenLayer1Size hiddenLayer2Size],trainFcn);
end
if numberOfHiddenLayers==1
    trainFcn = 'trainbr';
    hiddenLayer1Size = 8;
    net = fitnet(hiddenLayer1Size,trainFcn);
end
% Selection of internal transfer functions
%     compet - Competitive transfer function.
%     elliotsig - Elliot sigmoid transfer function.
%     hardlim - Positive hard limit transfer function.
%     hardlims - Symmetric hard limit transfer function.
%     logsig - Logarithmic sigmoid transfer function.
%     netinv - Inverse transfer function.
%     poslin - Positive linear transfer function.
%     purelin - Linear transfer function.
%     radbas - Radial basis transfer function.
%     radbasn - Radial basis normalized transfer function.
%     satlin - Positive saturating linear transfer function.
%     satlins - Symmetric saturating linear transfer function.
%     softmax - Soft max transfer function.
%     tansig - Symmetric sigmoid transfer function.
%     tribas - Triangular basis transfer function.
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
if numberOfHiddenLayers==2
    net.layers{3}.transferFcn = 'tansig';
end
% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
% mapminmax: y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin;
net.input.processFcns = {'mapminmax'};
net.output.processFcns = {'mapminmax'};
%net.input.processFcns = {'removeconstantrows','mapminmax'};
%net.output.processFcns = {'removeconstantrows','mapminmax'};
if loopmode==true
if nubmerOfOutputs==3
    e=[100;100;100];
    testparameter=max(abs(e(3,:)));
    maxx=23;
    testparameter2=1;
    maxxx=1.1;
elseif nubmerOfOutputs==2
    e=[100;100];
    testparameter=max(abs(e(2,:)));
    testparameter2=max(abs(e(1,:)));
    maxx=6.5;
    maxxx=1.5;
elseif nubmerOfOutputs==1;
    e=100;
    testparameter=max(abs(e));
    maxx=23;
    testparameter2=1;
    maxxx=1.1;
end
while testparameter>=maxx | testparameter2>=maxxx
% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 20/100;
if trainFcn == 'trainbr'
   net.divideParam.trainRatio = 80/100;
   net.divideParam.valRatio = 0/100;
   net.divideParam.testRatio = 20/100;
end
% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error
% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist','plotregression'};

% Train the Network
[net,tr] = train(net,x,t);
% Test the Network
y = net(x);
e = gsubtract(t,y);

    if nubmerOfOutputs==3
        testparameter=max(abs(e(3,:)))
    elseif nubmerOfOutputs==2
        testparameter=max(abs(e(2,:)));
        testparameter2=max(abs(e(1,:)));
        [testparameter testparameter2]
    elseif nubmerOfOutputs==1;
    testparameter=max(abs(e))
    end
end
else
% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 20/100;
if trainFcn == 'trainbr'
   net.divideParam.trainRatio = 80/100;
   net.divideParam.valRatio = 0/100;
   net.divideParam.testRatio = 20/100;
end
% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error
% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist','plotregression'};

% Train the Network
[net,tr] = train(net,x,t);
% Test the Network
y = net(x);
e = gsubtract(t,y);
end
% performance = perform(net,t,y);
% Recalculate Training, Validation and Test Performance
% What is an Epoch? In terms of artificial neural networks, an epoch refers to one cycle through the full training dataset.
% trainTargets = t .* tr.trainMask{1};
% valTargets = t .* tr.valMask{1};
% testTargets = t .* tr.testMask{1};
% trainPerformance = perform(net,trainTargets,y);
% valPerformance = perform(net,valTargets,y);
% testPerformance = perform(net,testTargets,y);


%test
number=[];
number=[1:length(t)];

for ii=1:nubmerOfOutputs

if nubmerOfOutputs==1
TN= 'Efficiency';
% TN Title Name
else
if ii==1
TN= 'Evaporation Rate';
elseif ii==2
TN= 'Top Temperature';
elseif ii==3
TN= 'Efficiency';
end
end
figure
subplot(1,2,1)
plot(number,sort(t(ii,:)),'O--','LineWidth',2)
hold on
plot(number,sort(y(ii,:)),'rX-','LineWidth',2)
legend('Real Data', 'Predicted Data')
title(TN)
subplot(1,2,2)
bar(number,sort(abs(e(ii,:))),'r')
legend('Error')
title([TN ' Error'])

figure
subplot(2,1,1)
plot(number,t(ii,:),'O--','LineWidth',2)
hold on
plot(number,y(ii,:),'rX-','LineWidth',2)
legend('Real Data', 'Predicted Data')
title(TN)
subplot(2,1,2)
bar(number,e(ii,:),'b')
legend('Error')
title([TN ' Error'])
end

% optimization

