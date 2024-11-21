% 清除環境變數並關閉所有圖形
clear; close all; clc;

% 1. 資料導入
df_train = readtable('train.csv');
original = readtable('ObesityDataSet.csv');
df_test = readtable('test.csv');

% 2. 資料預處理
% 2.1 移除 'id' 欄位
if ismember('id', df_train.Properties.VariableNames)
    df_train.id = [];
end

if ismember('id', original.Properties.VariableNames)
    original.id = [];
end

if ismember('id', df_test.Properties.VariableNames)
    df_test.id = [];
end

% 2.2 合併訓練資料和原始資料並移除重複項
% 找出所有變數名稱的並集
allVars = union(df_train.Properties.VariableNames, original.Properties.VariableNames);

% 為 df_train 添加缺少的變數
missingInTrain = setdiff(allVars, df_train.Properties.VariableNames);
for i = 1:length(missingInTrain)
    varName = missingInTrain{i};
    % 根據變數類型填充缺失值
    % 假設缺失變數為數值型，若為分類型，請調整為 categorical(NaN)
    df_train.(varName) = NaN(height(df_train), 1);
end

% 為 original 添加缺少的變數
missingInOriginal = setdiff(allVars, original.Properties.VariableNames);
for i = 1:length(missingInOriginal)
    varName = missingInOriginal{i};
    % 根據變數類型填充缺失值
    % 假設缺失變數為數值型，若為分類型，請調整為 categorical(NaN)
    original.(varName) = NaN(height(original), 1);
end

% 重新排列變數順序，使兩個表格的變數順序一致
df_train = df_train(:, allVars);
original = original(:, allVars);

% 垂直合併
train = [df_train; original];

% 移除重複行
train = unique(train, 'rows');

% 2.3 檢查缺失值
missing_train = sum(ismissing(train));
missing_test = sum(ismissing(df_test));
disp('訓練集缺失值統計：');
disp(missing_train);
disp('測試集缺失值統計：');
disp(missing_test);

% 2.4 移除缺失值（如果有）
train = rmmissing(train);
df_test = rmmissing(df_test);

% 2.5 處理類別變數
% 找出資料中的類別變數
categoricalVars = varfun(@iscellstr, train, 'OutputFormat', 'uniform') | ...
                  varfun(@isstring, train, 'OutputFormat', 'uniform');
categoricalVars = train.Properties.VariableNames(categoricalVars);

% 將目標變數 'NObeyesdad' 從列表中移除
categoricalVars_predictors = setdiff(categoricalVars, {'NObeyesdad'});

% 將類別變數轉換為分類型別
for i = 1:length(categoricalVars)
    varName = categoricalVars{i};
    train.(varName) = categorical(train.(varName));
    if ismember(varName, df_test.Properties.VariableNames)
        df_test.(varName) = categorical(df_test.(varName), categories(train.(varName)));
    end
end

% 2.6 編碼類別變數並避免多重共線性
% 使用 One-Hot 編碼，並刪除每個類別變數的最後一個虛擬變數
for i = 1:length(categoricalVars_predictors)
    varName = categoricalVars_predictors{i};
    
    % 對訓練集進行 One-Hot 編碼
    dummyTrain = dummyvar(train.(varName));
    categories_var = categories(train.(varName));
    
    % 創建新的變數名稱，並刪除最後一個虛擬變數
    dummyVarNames = strcat(varName, '_', categories_var);
    dummyTrain = dummyTrain(:, 1:end-1); % 刪除最後一個虛擬變數
    dummyVarNames = dummyVarNames(1:end-1);
    
    % 將編碼後的變數添加到訓練資料表中
    train = [train, array2table(dummyTrain, 'VariableNames', dummyVarNames)];
    
    % 對測試集進行相同的處理
    if ismember(varName, df_test.Properties.VariableNames)
        dummyTest = dummyvar(df_test.(varName));
        dummyTest = dummyTest(:, 1:end-1); % 刪除最後一個虛擬變數
        dummyTestNames = dummyVarNames; % 使用訓練集的變數名稱
        df_test = [df_test, array2table(dummyTest, 'VariableNames', dummyTestNames)];
    end
    
    % 移除原始的類別變數
    train.(varName) = [];
    if ismember(varName, df_test.Properties.VariableNames)
        df_test.(varName) = [];
    end
end

% 2.7 處理目標變數
% 將目標變數轉換為數值型別
[Y, targetLevels] = grp2idx(train.NObeyesdad);
train.Y = Y;
train.NObeyesdad = []; % 移除原始目標變數

% 2.8 特徵縮放
% 對數值變數進行標準化，並排除 'Y'
numericVars = varfun(@isnumeric, train, 'OutputFormat', 'uniform');
numericVarsNames = train.Properties.VariableNames(numericVars);
numericVarsNames = setdiff(numericVarsNames, 'Y'); % 排除 'Y'

% 提取數值特徵
X = train(:, numericVarsNames);
mu = mean(table2array(X));
sigma = std(table2array(X));

% 避免除以零
sigma(sigma == 0) = 1;

% 標準化
X_standardized = array2table((table2array(X) - mu) ./ sigma, 'VariableNames', numericVarsNames);

% 更新特徵矩陣
train(:, numericVarsNames) = X_standardized;

% 對測試集進行相同的處理
X_test = df_test(:, numericVarsNames);
X_test_standardized = array2table((table2array(X_test) - mu) ./ sigma, 'VariableNames', numericVarsNames);
df_test(:, numericVarsNames) = X_test_standardized;

% 3. 特徵選擇與降維（選擇性步驟）
% 3.1 移除高度相關的特徵
corrMatrix = corr(table2array(X_standardized), 'Rows', 'complete');
threshold = 0.9;
[rows, cols] = find(abs(corrMatrix) > threshold & abs(corrMatrix) < 1);

featuresToRemove = unique(cols);
if ~isempty(featuresToRemove)
    featuresToRemoveNames = X.Properties.VariableNames(featuresToRemove);
    X_reduced = train(:, ~ismember(train.Properties.VariableNames, featuresToRemoveNames));
else
    X_reduced = train;
end

% 提取數值特徵縮減後的名稱，並排除 'Y'
numericVars_reduced = varfun(@isnumeric, X_reduced, 'OutputFormat', 'uniform');
numericVarsNames_reduced = X_reduced.Properties.VariableNames(numericVars_reduced);
numericVarsNames_reduced = setdiff(numericVarsNames_reduced, 'Y'); % 排除 'Y'

X_reduced = X_reduced(:, numericVarsNames_reduced);

% 4. 資料分割
% 將資料分為訓練集和測試集
cv = cvpartition(Y, 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);

X_train = X_reduced(idxTrain, :);
Y_train = Y(idxTrain);
X_test_final = X_reduced(idxTest, :);
Y_test_final = Y(idxTest);

% 5. 模型建立與評估
% 定義要使用的模型
models = {'Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression', 'KNN', 'Naive Bayes'};
numModels = length(models);

% 初始化評估指標
accuracy = zeros(numModels, 1);
precision = zeros(numModels, 1);
recall = zeros(numModels, 1);
f1_score = zeros(numModels, 1);
confMatrices = cell(numModels, 1);

% 交叉驗證設定
K = 5;
cv_partition = cvpartition(Y_train, 'KFold', K);

for m = 1:numModels
    modelName = models{m};
    fprintf('正在訓練模型：%s\n', modelName);
    
    % 初始化交叉驗證的評估指標
    cv_accuracy = zeros(K, 1);
    
    for k = 1:K
        idxTrain_cv = training(cv_partition, k);
        idxVal_cv = test(cv_partition, k);
        
        X_train_cv = X_train(idxTrain_cv, :);
        Y_train_cv = Y_train(idxTrain_cv);
        X_val_cv = X_train(idxVal_cv, :);
        Y_val_cv = Y_train(idxVal_cv);
        
        % 訓練模型
        switch modelName
            case 'Decision Tree'
                model = fitctree(X_train_cv, Y_train_cv);
            case 'Random Forest'
                model = TreeBagger(100, X_train_cv, Y_train_cv, 'Method', 'classification');
            case 'SVM'
                model = fitcecoc(X_train_cv, Y_train_cv);
            case 'Logistic Regression'
                % 使用 fitmnr 進行多分類邏輯迴歸
                model = fitmnr(table2array(X_train_cv), Y_train_cv);
            case 'KNN'
                model = fitcknn(X_train_cv, Y_train_cv, 'NumNeighbors', 5);
            case 'Naive Bayes'
                model = fitcnb(X_train_cv, Y_train_cv);
            otherwise
                error('未知的模型類型');
        end
        
        % 預測驗證集
        switch modelName
            case 'Random Forest'
                Y_pred_cv = str2double(predict(model, X_val_cv));
            case 'Logistic Regression'
                Y_pred_cv = predict(model, table2array(X_val_cv));
            otherwise
                Y_pred_cv = predict(model, X_val_cv);
        end
        
        % 計算準確率
        cv_accuracy(k) = sum(Y_pred_cv == Y_val_cv) / numel(Y_val_cv);
    end
    
    % 計算平均準確率
    accuracy(m) = mean(cv_accuracy);
    
    % 在測試集上評估
    % 使用整個訓練集重新訓練模型
    switch modelName
        case 'Decision Tree'
            model_final = fitctree(X_train, Y_train);
            Y_pred_test = predict(model_final, X_test_final);
        case 'Random Forest'
            model_final = TreeBagger(100, X_train, Y_train, 'Method', 'classification');
            Y_pred_test = str2double(predict(model_final, X_test_final));
        case 'SVM'
            model_final = fitcecoc(X_train, Y_train);
            Y_pred_test = predict(model_final, X_test_final);
        case 'Logistic Regression'
            model_final = fitmnr(table2array(X_train), Y_train);
            Y_pred_test = predict(model_final, table2array(X_test_final));
        case 'KNN'
            model_final = fitcknn(X_train, Y_train, 'NumNeighbors', 5);
            Y_pred_test = predict(model_final, X_test_final);
        case 'Naive Bayes'
            model_final = fitcnb(X_train, Y_train);
            Y_pred_test = predict(model_final, X_test_final);
    end
    
    % 計算混淆矩陣和評估指標
    confMat = confusionmat(Y_test_final, Y_pred_test);
    confMatrices{m} = confMat;
    
    % 計算 Precision, Recall, F1-score
    [precision(m), recall(m), f1_score(m)] = computeMetrics(confMat);
end

% 6. 結果總結
% 顯示各模型的評估結果
resultsTable = table(models', accuracy, precision, recall, f1_score, 'VariableNames', ...
    {'Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score'});

disp('各模型的評估結果：');
disp(resultsTable);

% 7. 繪製混淆矩陣
for m = 1:numModels
    figure;
    confMat = confMatrices{m};
    cm = confusionchart(confMat);
    cm.Title = ['Confusion Matrix - ', models{m}];
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';
    % 保存圖片
    saveas(gcf, ['Confusion_Matrix_', models{m}, '.png']);
    close(gcf);
end

% 8. 繪製特徵重要性（以隨機森林為例）
rfIndex = find(strcmp(models, 'Random Forest'));
if ~isempty(rfIndex)
    rfModel_final = TreeBagger(100, X_train, Y_train, 'Method', 'classification');
    feature_importance = rfModel_final.OOBPermutedPredictorDeltaError;
    feature_importance_df = table(X_train.Properties.VariableNames', feature_importance, ...
        'VariableNames', {'Feature', 'Importance'});
    feature_importance_df = sortrows(feature_importance_df, 'Importance', 'descend');
    
    figure;
    bar(feature_importance_df.Importance);
    title('Feature Importance from Random Forest');
    xlabel('Features');
    ylabel('Importance');
    set(gca, 'XTick', 1:length(feature_importance_df.Feature), 'XTickLabel', feature_importance_df.Feature, ...
        'XTickLabelRotation', 45);
    grid on;
    % 保存圖片
    saveas(gcf, 'Feature_Importance_Random_Forest.png');
    close(gcf);
end

% 9. 結論與建議
% 根據評估結果，選擇性能最佳的模型，並討論可能的改進方向。

% === 定義計算評估指標的函數 ===
function [precision, recall, f1_score] = computeMetrics(confMat)
    % 假設為多分類問題，計算宏平均的 Precision、Recall、F1-score
    numClasses = size(confMat, 1);
    precision_per_class = zeros(numClasses, 1);
    recall_per_class = zeros(numClasses, 1);
    
    for i = 1:numClasses
        TP = confMat(i, i);
        FP = sum(confMat(:, i)) - TP;
        FN = sum(confMat(i, :)) - TP;
        precision_per_class(i) = TP / (TP + FP + eps);
        recall_per_class(i) = TP / (TP + FN + eps);
    end
    
    precision = mean(precision_per_class);
    recall = mean(recall_per_class);
    f1_score = 2 * (precision * recall) / (precision + recall + eps);
end
