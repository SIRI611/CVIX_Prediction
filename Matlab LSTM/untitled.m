clear clc:
t = data2;
y=CVIX;
data = [BrentFuturesSettlement,creditSpread,HML_total,interestRate,oneYearBondYield,pbOfSZ50,peOfSZ50,RiskPremium_total,shenwanHouseMarket,SMB_total,SZ50,tenYearBondYield,termSpread,turnoverOfSZ50,USD,volatilityOf50ETFOptions,weightedAverageRepurchaseRate,weightedAverageGoldPrice,y];
%plot(t,y);
%ylabel("y");

numTimeStepsTrain = floor(0.9*numel(y));
dataTrain = data(1:numTimeStepsTrain,:);
dataTest = data(numTimeStepsTrain+1:end,:);

%归一化
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - ones(length(dataTrain( :,1)),1)*mu) ./ (ones(length(dataTrain(:,1)),1)*sig);


XTrain = dataTrainStandardized(:,1:18)';
YTrain = dataTrainStandardized(:,19)';

layers = [
    sequenceInputLayer(18,"Name","input")
    lstmLayer(128,"Name","lstm")
    dropoutLayer(0.5,"Name","drop")
    fullyConnectedLayer(1,"Name","fc")
    regressionLayer("Name","regressionoutput")];

options = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress'); 

net = trainNetwork(XTrain,YTrain,layers,options); 

dataTestStandardized = (dataTest - ones(length(dataTest( :,1)),1)*mu) ./ (ones(length(dataTest(:,1)),1)*sig);
XTest = dataTestStandardized(:,1:18)';

net = predictAndUpdateState(net,XTrain);

numTimeStepsTest = numel(XTest(1,:));
YPred = [];
for i = 1:numTimeStepsTest
    [net,YPred(i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end

YPred = sig(19)*YPred + mu(19);

pathout="C:\\Users\\siri\\OneDrive\\大创VIX指数\\YPred(2).xlsx"
xlswrite(pathout,YPred.');

%绘图
idx = (numTimeStepsTrain+1):(numTimeStepsTrain+numTimeStepsTest);
figure
set(gcf,'position',[1,1,1500,1000])
set(gca,'position',[0.1,0.1,0.8,4])
subplot(2,1,1)
plot(data(1:end,19))
hold on
plot(idx,YPred,'-.k',LineWidth=1.5)
hold off;
xlabel('id')
ylabel('y')

subplot(2,1,2)
plot(data(1:end,19))
hold on
plot(idx , YPred,'-.^k')
hold off
xlabel('id')
ylabel('y')
xlim([1512,1680])
title('Forecast')


