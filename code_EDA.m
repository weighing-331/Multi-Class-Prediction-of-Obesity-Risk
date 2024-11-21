% 1.1 導入資料
df_train = readtable('train.csv');
original = readtable('ObesityDataSet.csv');
df_test = readtable('test.csv');

% 檢查資料的形狀
fprintf('訓練集包含 %d 行和 %d 列\n', size(df_train, 1), size(df_train, 2));
fprintf('測試集包含 %d 行和 %d 列\n', size(df_test, 1), size(df_test, 2));

% 將描述統計資訊保存為 CSV 文件
summaryData = summary(df_train);
% 將結構體轉換為表格以便保存為 CSV
summaryTable = struct2table(summaryData);
writetable(summaryTable, 'output.csv');

% 1.3 快速預覽
disp('訓練集預覽:');
head(df_train)

disp('測試集預覽:');
head(df_test)

% 1.4 資料摘要
% 移除 'id' 列
df_train_no_id = removevars(df_train, {'id'});
summary(df_train_no_id)

% 2. 探索性資料分析
% 定義保存圖片的目錄
saveDir = 'plot';

% 如果目錄不存在，則創建目錄
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

% 計算 BMI 並添加到資料集中
df_train.BMI = df_train.Weight ./ (df_train.Height .^ 2);

% 顯示目標變數的分佈
showplot(df_train, 'NObeyesdad', saveDir);

% 顯示其他類別變數的分佈
categoricalVars = {'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', ...
    'SMOKE', 'SCC', 'CALC', 'MTRANS'};
for i = 1:length(categoricalVars)
    showplot(df_train, categoricalVars{i}, saveDir);
end

% 數值變數的分佈
numericVars = {'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI'};
for i = 1:length(numericVars)
    figure;
    histogram(df_train.(numericVars{i}));
    title(['Distribution of ', numericVars{i}]);
    xlabel(numericVars{i});
    ylabel('Frequency');
    grid on;

    % 保存圖片
    saveas(gcf, fullfile(saveDir, [numericVars{i}, '_distribution.png']));
    close(gcf);
end

% 與目標變數的關係可視化
% 繪製數值變數與目標變數的箱型圖
for i = 1:length(numericVars)
    figure;
    boxplot(df_train.(numericVars{i}), df_train.NObeyesdad);
    title([numericVars{i}, ' vs NObeyesdad']);
    xlabel('NObeyesdad');
    ylabel(numericVars{i});
    xtickangle(45);

    % 保存圖片
    saveas(gcf, fullfile(saveDir, [numericVars{i}, '_vs_NObeyesdad.png']));
    close(gcf);
end

% 計算每個類別的 BMI 統計資訊
groupStats = grpstats(df_train, 'NObeyesdad', {'mean', 'std'}, 'DataVars', 'BMI');
disp(groupStats);

% 計算特徵之間的相關性
% 將類別變數編碼為數值
df_numeric = df_train_no_id;
categoricalVarsNumeric = varfun(@iscellstr, df_numeric, 'OutputFormat', 'uniform');
for i = find(categoricalVarsNumeric)
    df_numeric.(df_numeric.Properties.VariableNames{i}) = grp2idx(df_numeric.(df_numeric.Properties.VariableNames{i}));
end

% 計算相關係數矩陣
corrMatrix = corr(table2array(df_numeric), 'Rows', 'complete');

% 繪製相關係數矩陣的熱圖
figure;
heatmap(df_numeric.Properties.VariableNames, df_numeric.Properties.VariableNames, corrMatrix, 'Colormap', jet, 'ColorLimits', [-1, 1]);
title('Correlation Matrix');

% 保存圖片
saveas(gcf, fullfile(saveDir, 'Correlation_Matrix.png'));
close(gcf);

% 3. 資料預處理
% 檢查缺失值
missingValues = sum(ismissing(df_train));
disp('每個欄位的缺失值數量：');
disp(missingValues);

% 檢查重複值
numDuplicates = height(df_train) - height(unique(df_train));
fprintf('訓練集共有 %d 個重複樣本。\n', numDuplicates);

% 顯示統計結果

numericVars = {'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI'};
numericData = df_train(:, numericVars);


meanValues = varfun(@mean, numericData, 'OutputFormat', 'uniform');
stdValues = varfun(@std, numericData, 'OutputFormat', 'uniform');
minValues = varfun(@min, numericData, 'OutputFormat', 'uniform');
maxValues = varfun(@max, numericData, 'OutputFormat', 'uniform');
medianValues = varfun(@median, numericData, 'OutputFormat', 'uniform');


summaryData = [meanValues; stdValues; minValues; maxValues; medianValues];
summaryTable = array2table(summaryData, 'VariableNames', numericVars);
summaryTable.Properties.RowNames = {'Mean', 'StdDev', 'Min', 'Max', 'Median'};


disp('数值变量的统计摘要：');
disp(summaryTable);

% 3.3 視覺化變數之間的關係
% 繪製體重與身高的散佈圖，按肥胖程度著色
figure;
gscatter(df_train.Height, df_train.Weight, df_train.NObeyesdad);
xlabel('Height');
ylabel('Weight');
title('Weight vs Height by Obesity Level');
legend('Location', 'bestoutside');
grid on;

% 保存圖片
saveas(gcf, fullfile(saveDir, 'Weight_vs_Height_by_Obesity_Level.png'));
close(gcf);

% 數值變數的成對散佈圖
selectedVars = {'Age', 'Height', 'Weight', 'BMI', 'FAF'};
figure;
plotmatrix(table2array(df_train(:, selectedVars)));
title('Pairwise Scatter Plots');

% 保存圖片
saveas(gcf, fullfile(saveDir, 'Pairwise_Scatter_Plots.png'));
close(gcf);

% 3.4 帶有註解的相關矩陣熱圖
% 計算數值變數的相關係數矩陣
corrMatrix = corr(table2array(df_train(:, numericVars)), 'Rows', 'complete');

% 創建帶有相關值註解的熱圖
figure;
h = heatmap(numericVars, numericVars, corrMatrix, 'Colormap', jet, 'ColorLimits', [-1, 1]);
h.CellLabelFormat = '%.2f';
title('Correlation Matrix Heatmap');

% 保存圖片
saveas(gcf, fullfile(saveDir, 'Correlation_Matrix_Annotated.png'));
close(gcf);

% 3.5 按性別分析 BMI 分佈
figure;
boxplot(df_train.BMI, df_train.Gender);
xlabel('Gender');
ylabel('BMI');
title('BMI Distribution by Gender');

% 保存圖片
saveas(gcf, fullfile(saveDir, 'BMI_by_Gender.png'));
close(gcf);

% 3.6 類別便亮


df_train.Gender = categorical(df_train.Gender);
df_train.NObeyesdad = categorical(df_train.NObeyesdad);


[genderObesityTable, genders, obesityLevels] = crosstab(df_train.Gender, df_train.NObeyesdad);


genders = cellstr(genders);
obesityLevels = cellstr(obesityLevels);


genders = matlab.lang.makeValidName(genders, 'ReplacementStyle', 'delete');
obesityLevels = matlab.lang.makeValidName(obesityLevels, 'ReplacementStyle', 'delete');


crossTabTable = array2table(genderObesityTable, 'VariableNames', obesityLevels, 'RowNames', genders);
disp('性别与肥胖程度的交叉列联表：');
disp(crossTabTable);

figure;
bar(categorical(genders), genderObesityTable, 'stacked');
xlabel('Gender');
ylabel('Count');
title('Obesity Level Distribution by Gender');
legend(obesityLevels, 'Location', 'bestoutside');


saveas(gcf, fullfile(saveDir, 'Obesity_Level_by_Gender.png'));
close(gcf);


% 3.7 使用隨機森林進行特徵重要性分析
% 將類別變數編碼為數值
df_encoded = df_train;
categoricalVars = {'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', ...
    'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'};
for i = 1:length(categoricalVars)
    varName = categoricalVars{i};
    df_encoded.(varName) = grp2idx(categorical(df_encoded.(varName)));
end

% 準備資料進行建模
X = table2array(df_encoded(:, 1:end-1));
Y = df_encoded.NObeyesdad;

% 訓練隨機森林模型
rng('default');
rfModel = TreeBagger(100, X, Y, 'OOBPredictorImportance', 'on');

% 獲取特徵重要性
importance = rfModel.OOBPermutedPredictorDeltaError;

% 繪製特徵重要性圖
[sortedImportance, idx] = sort(importance, 'descend');
figure;
bar(sortedImportance);
title('Feature Importance from Random Forest');
xlabel('Features');
ylabel('Importance');
xticks(1:length(idx));
xticklabels(df_encoded.Properties.VariableNames(idx));
xtickangle(45);

% 保存圖片
saveas(gcf, fullfile(saveDir, 'Feature_Importance_RF.png'));
close(gcf);

% 3.8 主成分分析（PCA）
% 標準化數值變數
X_numeric = table2array(df_train(:, numericVars));
X_std = (X_numeric - mean(X_numeric)) ./ std(X_numeric);

% 執行 PCA
[coeff, score, latent, tsquared, explained] = pca(X_std);

% 繪製累積解釋變異量圖
figure;
pareto(explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('PCA Explained Variance');

% 保存圖片
saveas(gcf, fullfile(saveDir, 'PCA_Explained_Variance.png'));
close(gcf);

% 繪製前兩個主成分的散佈圖
figure;
gscatter(score(:,1), score(:,2), df_train.NObeyesdad);
xlabel('PC1');v
ylabel('PC2');
title('PCA - First Two Principal Components');
legend('Location', 'bestoutside');

% 保存圖片
saveas(gcf, fullfile(saveDir, 'PCA_Scatter_Plot.png'));
close(gcf);

