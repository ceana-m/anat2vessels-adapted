%% Rank variability per subject
%% compute ranks per visit
rankData = struct();

for f = 1:length(features)
    feat = features(f);
    M = data.(feat);
    
    % ranks per visit (higher value = better rank)
    R = nan(size(M));
    
    for j = 1:3
        R(:,j) = tiedrank(-M(:,j)); % descending rank
    end
    
    rankData.(feat) = R;
end

%% compute rank variability per subject
rank_var_table = table();

for f = 1:length(features)
    feat = features(f);
    R = rankData.(feat);
    
    rank_sd = std(R, 0, 2, 'omitnan'); % per subject
    
    rank_var_table.subject = subjects;
    rank_var_table.(feat) = rank_sd;
    
    if f == 1
        allRankSD = rank_sd;
    else
        allRankSD = [allRankSD rank_sd];
    end
end

%% summarize per feature
rank_summary = table();

for f = 1:length(features)
    feat = features(f);
    
    R = rankData.(feat);
    rank_sd = std(R, 0, 2, 'omitnan');
    
    rank_summary.feature(f) = feat;
    rank_summary.mean_rank_SD(f) = mean(rank_sd,'omitnan');
    rank_summary.median_rank_SD(f) = median(rank_sd,'omitnan');
    rank_summary.max_rank_SD(f) = max(rank_sd);
end

disp(rank_summary)

%% subject level heat map
figure;

heatmap( ...
    string(subjects), ...
    rank_summary.feature, ...
    allRankSD', Interpreter="none");

title("Subject-wise Rank Variability");
xlabel("Subject");
ylabel("Feature");
