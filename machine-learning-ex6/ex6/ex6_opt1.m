%% Initialization
clear ; close all; clc;

%load('spamTrain.mat');

%fprintf('\nTraining Linear SVM (Spam Classification)\n')
%fprintf('(this may take 1 to 2 minutes) ...\n')

%C = 0.1;
%model = svmTrain(X, y, C, @linearKernel);
load pre-load-model.mat;
%save pre-load-model.mat model;

%files = {'emailSample1.txt', 'emailSample2.txt', 'spamSample1.txt', 'spamSample2.txt'};
files = {'testEmail.txt', 'testEmail2.txt'};

for i = 1:length(files)

  file_contents = readFile(files{i});
  word_indices  = processEmail(file_contents);
  features      = emailFeatures(word_indices);
  p = svmPredict(model, features);
  
  fprintf('<<< File %s Predict: %i >>>\n', files{i}, p);
end
