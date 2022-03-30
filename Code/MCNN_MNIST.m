clear all; clc; 
% https://kr.mathworks.com/help/deeplearning/examples/create-simple-deep-learning-network-for-classification.html

%������ �з��� ���� �н� ��Ʈ��ũ �����
%�� ���������� ���� �н� �з������� ������ ������� �Ű���� ����� �Ʒý�Ű�� ����� �����ݴϴ�. 
%������� �Ű���� ���� �н� �о��� �ʼ� ���μ�, Ư�� �̹��� �νĿ� �����մϴ�.
%�� ���������� ������ �����ϴ� ����� �����ݴϴ�.

%�̹��� �����͸� �ҷ��ͼ�.
%��Ʈ��ũ ��Ű��ó�� ����.
%�Ʒ� �ɼ��� ����.
%��Ʈ��ũ�� �Ʒ�.
%���ο� �������� ���̺��� �����ϰ� �з� ��Ȯ���� ���.


%�̹��� �����͸� �ҷ��ͼ� ���캸��
% C:\Program Files\MATLAB\R2019a\toolbox\nnet\nndemos\nndatasets\DigitDataset
%���� ���� �����͸� �̹��� ����������ҷμ� �ҷ��ɴϴ�. 
%imageDatastore�� ���� �̸��� �������� �̹����� �ڵ����� ���̺��� �����ϰ� �����͸� ImageDatastore ��ü�� ����. 
%�̹��� ����������Ҹ� ����ϸ� �޸𸮿� ���� �� ���� �����͸� �����Ͽ� �ٷ��� �̹��� �����͸� ������ �� �ְ� ������� �Ű�� �Ʒ� �߿� �̹��� ��ġ�� ȿ�������� �о� ���� �� �ִ�.
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


%�Ʒ� ��Ʈ�� ���� ��Ʈ �����ϱ�
%splitEachLabel�� ����������� digitData�� 2���� ���ο� ����������� trainDigitData�� valDigitData�� ����.

numTrainFiles = 0.8;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');


%��Ʈ��ũ ��Ű��ó �����ϱ�
%������� �Ű�� ��Ű��ó�� ����



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



%�Ʒ� �ɼ� �����ϱ�
%��Ʈ��ũ ������ ������ �������� �Ʒ� �ɼ��� ����. 
%SGDM(Stochastic Gradient Descent with Momentum: ������� ����� Ȯ���� ����ϰ���)�� ����Ͽ� �ʱ� �н���(�����Ͻ�)�� ��Ʈ��ũ�� �Ʒý�ŵ�ϴ�. 
%�ִ� Epoch Ƚ���� 10�� �����մϴ�. Epoch 1ȸ�� ��ü �Ʒ� ������ ��Ʈ�� ���� �ϳ��� ������ �Ʒ� �ֱ⸦ �ǹ�. 
%���� �����Ϳ� ���� �󵵸� �����Ͽ� �Ʒ� �߿� ��Ʈ��ũ ��Ȯ���� ����͸��մϴ�. �Ʒ� ���� �Ʒ� �����Ϳ� ���� �����͸� 1ȸ �����ϴ�. 
%�Ʒ� �����Ϳ� ���� ��Ʈ��ũ�� �Ʒõǰ�, �Ʒ� �߿� ��Ģ���� �������� ���� �����Ϳ� ���� ��Ȯ���� ���. 
%���� �����ʹ� ��Ʈ��ũ ����ġ�� ������Ʈ�ϴ� �� ������ �ʽ��ϴ�. 
%weight update���ؼ� ������ ���ϱ� �� �����ϴ�

% sgdm=������� ����� Ȯ���� ����ϰ����� �Ʒ� �ɼ��Դϴ�. �н��� ����, L2 ����ȭ ����, �̴� ��ġ ���� �ش�.
%�� step ���� �Ϻ� DATA (mini-batch) �� ���ؼ��� gradient �� ����ϴ� ����� Stochastic Gradient Descent ��� �Ѵ�. 
%SGD �� ���� �ð��� ������ gradient �� ����� �� �����Ƿ� ���� �ӵ��� ���̰� Ư�� local minimum �� �� ������ �ʴ´ٴ� ������ �ִ�.
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

%�Ʒ� �����͸� ����Ͽ� ��Ʈ��ũ �Ʒý�Ű��
%layers�� ���� ���ǵ� ��Ű��ó, �Ʒ� ������ �� �Ʒ� �ɼ��� ����Ͽ� ��Ʈ��ũ�� �Ʒý�ŵ�ϴ�.  
%�Ʒ� ���� ��Ȳ �÷Կ� �̴� ��ġ�� �ս� �� ��Ȯ���� ������ �ս� �� ��Ȯ���� ǥ�õ˴ϴ�. 
%�Ʒ� ���� ��Ȳ �÷Կ� ���� �ڼ��� ������ ���� �н� �Ʒ� ���� ��Ȳ ����͸��ϱ� �׸��� ����. 
%�ս��� ���� ��Ʈ���� �ս��Դϴ�. ��Ȯ���� ��Ʈ��ũ�� �ùٸ��� �з��� �̹����� ����.

list1 = [];
for i = 1:30
    
    net = trainNetwork(imdsTrain,layers,options);

    gpuDevice;

    %���� �̹����� �з��ϰ� ��Ȯ�� ����ϱ�
    %�Ʒõ� ��Ʈ��ũ�� ����Ͽ� ���� �������� ���̺��� �����ϰ� ���� ���� ��Ȯ���� ���. 
    %��Ȯ���� ��Ʈ��ũ�� �ùٸ��� �����ϴ� ���̺��� ����. 
    %���⼭�� ������ ���̺��� 99% �̻��� ���� ��Ʈ�� ��¥ ���̺�� ��ġ.
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

