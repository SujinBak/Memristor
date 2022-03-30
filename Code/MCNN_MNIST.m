clear all; clc; 
% https://kr.mathworks.com/help/deeplearning/examples/create-simple-deep-learning-network-for-classification.html

%간단한 분류용 심층 학습 네트워크 만들기
%이 예제에서는 심층 학습 분류용으로 간단한 컨벌루션 신경망을 만들고 훈련시키는 방법을 보여줍니다. 
%컨벌루션 신경망은 심층 학습 분야의 필수 툴로서, 특히 이미지 인식에 적합합니다.
%이 예제에서는 다음을 수행하는 방법을 보여줍니다.

%이미지 데이터를 불러와서.
%네트워크 아키텍처를 정의.
%훈련 옵션을 지정.
%네트워크를 훈련.
%새로운 데이터의 레이블을 예측하고 분류 정확도를 계산.


%이미지 데이터를 불러와서 살펴보기
% C:\Program Files\MATLAB\R2019a\toolbox\nnet\nndemos\nndatasets\DigitDataset
%샘플 숫자 데이터를 이미지 데이터저장소로서 불러옵니다. 
%imageDatastore는 폴더 이름을 기준으로 이미지에 자동으로 레이블을 지정하고 데이터를 ImageDatastore 객체로 저장. 
%이미지 데이터저장소를 사용하면 메모리에 담을 수 없는 데이터를 포함하여 다량의 이미지 데이터를 저장할 수 있고 컨벌루션 신경망 훈련 중에 이미지 배치를 효율적으로 읽어 들일 수 있다.
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


%훈련 세트와 검증 세트 지정하기
%splitEachLabel은 데이터저장소 digitData를 2개의 새로운 데이터저장소 trainDigitData와 valDigitData로 분할.

numTrainFiles = 0.8;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');


%네트워크 아키텍처 정의하기
%컨벌루션 신경망 아키텍처를 정의



layers = [
    imageInputLayer([28 28 1])
    
    %convolution2dLayer(3,8,'Padding','same')
    convolution2dLayer(3,8,'Padding','same',...
    'WeightsInitializer', @(sz) rand(sz) *0.0000367, ...
    'BiasInitializer', @(sz) rand(sz) * 0.1);
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];



%훈련 옵션 지정하기
%네트워크 구조를 정의한 다음에는 훈련 옵션을 지정. 
%SGDM(Stochastic Gradient Descent with Momentum: 모멘텀을 사용한 확률적 경사하강법)을 사용하여 초기 학습률(컨덕턴스)로 네트워크를 훈련시킵니다. 
%최대 Epoch 횟수를 10로 설정합니다. Epoch 1회는 전체 훈련 데이터 세트에 대한 하나의 완전한 훈련 주기를 의미. 
%검증 데이터와 검증 빈도를 지정하여 훈련 중에 네트워크 정확도를 모니터링합니다. 훈련 전에 훈련 데이터와 검증 데이터를 1회 섞습니다. 
%훈련 데이터에 대해 네트워크가 훈련되고, 훈련 중에 규칙적인 간격으로 검증 데이터에 대한 정확도가 계산. 
%검증 데이터는 네트워크 가중치를 업데이트하는 데 사용되지 않습니다. 
%weight update안해서 선형성 비교하기 더 용이하다

% sgdm=모멘텀을 사용한 확률적 경사하강법의 훈련 옵션입니다. 학습률 정보, L2 정규화 인자, 미니 배치 등이 해당.
%매 step 마다 일부 DATA (mini-batch) 에 대해서만 gradient 를 계산하는 방식을 Stochastic Gradient Descent 라고 한다. 
%SGD 는 같은 시간에 여러번 gradient 를 계산할 수 있으므로 빠른 속도를 보이고 특히 local minimum 에 잘 빠지지 않는다는 장점이 있다.
%L2 regulation
%5 fold cross validation
  % 'LearnRateSchedule','piecewise', ...
  size=[8];
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',1, ...
    'Shuffle','once', ...
    'MiniBatchSize', size, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',64, ...
    'Verbose',true, ...
    'Plots','training-progress');

%훈련 데이터를 사용하여 네트워크 훈련시키기
%layers에 의해 정의된 아키텍처, 훈련 데이터 및 훈련 옵션을 사용하여 네트워크를 훈련시킵니다.  
%훈련 진행 상황 플롯에 미니 배치의 손실 및 정확도와 검증의 손실 및 정확도가 표시됩니다. 
%훈련 진행 상황 플롯에 대한 자세한 내용은 심층 학습 훈련 진행 상황 모니터링하기 항목을 참조. 
%손실은 교차 엔트로피 손실입니다. 정확도는 네트워크가 올바르게 분류한 이미지의 비율.

list1 = [];
for i = 1:30
    
    net = trainNetwork(imdsTrain,layers,options);

    gpuDevice;

    %검증 이미지를 분류하고 정확도 계산하기
    %훈련된 네트워크를 사용하여 검증 데이터의 레이블을 예측하고 최종 검증 정확도를 계산. 
    %정확도는 네트워크가 올바르게 예측하는 레이블의 비율. 
    %여기서는 예측된 레이블의 99% 이상이 검증 세트의 진짜 레이블과 일치.
    YPred = classify(net,imdsValidation);
    YValidation = imdsValidation.Labels;

    accuracy = sum(YPred == YValidation)/numel(YValidation);
   list1(i) = accuracy;
    plotconfusion(YPred,YValidation);
    fig = gcf;
    fig.Color = 'white';
    fig.InvertHardcopy = 'off';
    
%     C = confusionmat(YPred,YValidation);
%     imwrite(C, "confusion" +num2str(i)+ ".jpg");

    %data.Category = categorical(YPred);

    
    %histogram(YPred);
    %xlabel("Accuracy")
    %ylabel("Frequency")
    %title("Class Distribution")
    
end

