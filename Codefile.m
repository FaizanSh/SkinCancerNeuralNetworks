%% Importing data from CSV. Null Elements are already removed manually
filename = 'E:\Matlab-NNFS\cancer_data.csv';
delimiter = ',';
data = importdata(filename,delimiter);

%% Software attributes
trainingOnPercentage=70;
MaxEpochs=10000;
LearningRate=0.02;
Neurons=10;
PerformanceGoal=0.02;
StartWeights=0.3232;

%% finding data size using matlab's 'size' method with require demention as a parameter. here dem 1 is no of rows which are 699
dataTotalSize=size(data,1);

%% Setting Percentage Variables
spareDataPercentage=100-trainingOnPercentage;

%% data sorting with respect to output so benign get separated from milignant
data=sortrows(data,11);

%% Seperating relevent data in to another matrix
input=data(:,2:10);


%% As data have in total 16 places in column 7 where value is NULL, so we are taking mean value of column (benigns and milignants separately) that has NULL value. Mean of benigns is 2 and milignants is 7.

%finding counts first
benignsCount=0;
for i=1:dataTotalSize
    if data(i,11)==2
        benignsCount=benignsCount+1;
    end
end

malignantsCount=dataTotalSize-benignsCount;

%Setting up the mean value finder
count=0;
sum=0;
meanvalue = zeros(1,size(input,2));
for i=1:size(input,2)
    for j=(1:benignsCount)
        if  input(j,i)~=0
            sum = sum + input(j,i);
            count = count + 1;
        else
            
        end
    end
    meanvalue(i) = sum/count;
end

for i=1:size(input,2)
    for j=(1:benignsCount)
        if input(j,i)~=0
        else
            input(j,i)= ceil(meanvalue(i));
        end
    end
end

meanvalue = zeros(1,size(input,2));
sum =0;
count=0;
for i=1:size(input,2)
    for j=(benignsCount+1:dataTotalSize)
        if input(j,i)~=0
            sum = sum + input(j,i);
            count = count + 1;
        else
        end
    end
    meanvalue(i) = sum/count;
end

for i=1:size(input,2)
    for j=(benignsCount+1:dataTotalSize)
        if input(j,i)~=0
            
        else
            input(j,i)= ceil(meanvalue(i));
        end
    end
end


%% seperating malignants and benigns
benigns=(input(1:benignsCount,:));
malignants=(input(benignsCount+1:dataTotalSize,:));


%% percentage calculation
trainingBenignCount=ceil((benignsCount*trainingOnPercentage)/100);
trainingMalignantCount=ceil((malignantsCount*trainingOnPercentage)/100);

%% preparing Training and validation Data Inputs
trainingInputData = [benigns(1:trainingBenignCount,:) 
    malignants(1:trainingMalignantCount,:)];
validationInputData =  [benigns( trainingBenignCount+1: benignsCount,:)
    malignants(trainingMalignantCount+1: malignantsCount,:)];

%% Calculating size
trainingInputDataSize=size(trainingInputData,1);
validationInputDataSize=size(validationInputData,1);


%% preparing Training and validation Output Data
trainingOutputData=zeros(trainingInputDataSize,2);
for i=(1:trainingInputDataSize)
    if (i<=trainingBenignCount)
        trainingOutputData(i,1)=1;
    else
        trainingOutputData(i,2)=1;
    end
end

validationOutputData=zeros(validationInputDataSize,2);
for i=(1:validationInputDataSize)
    if (i<= benignsCount-trainingBenignCount)
        validationOutputData(i,1)=1;
    else
        validationOutputData(i,2)=1;
    end
end

%% Building feedforward Net
net = feedforwardnet([1000,100,10],'traingd');

net.layers{1}.transferFcn = 'tansig'; %poslin %logsig %tansig(default)
net.divideFcn='dividetrain';
net.trainParam.lr=LearningRate;
net.trainParam.epochs=1; %because we want to change it after one iteration
net.trainParam.goal = PerformanceGoal;
net.trainParam.max_fail= 10;

[net] = train(net,trainingInputData',trainingOutputData');
weights = getwb(net);

for i=1:size(weights,1)
    weights(i,1)=StartWeights;% changing waits
end
%seting weights again
net = setwb(net,weights);
net.trainParam.epochs=MaxEpochs;
[net,tr] = train(net,trainingInputData',trainingOutputData');
weights1 = getwb(net);
%% Testing data via sim 
testing = sim(net,validationInputData');
%% checking data with expected solution
count=0;
testing=testing';
for i=1:validationInputDataSize
    [a, b]=max(testing(i,:));
    %a batata hay kya berra hay or b main uska index aata hay
    if validationOutputData(i,b)==1
        count=count+1;
    end
end
%% finding accuracy
accuracy=(count/validationInputDataSize)*100;
%% display
disp(accuracy);