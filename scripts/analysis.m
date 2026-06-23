clear all; clc
%%
T = readtable('features.csv');
%% Extract subject and visit

parts = split(string(T.sub_id), '_');

T.visit = parts(:,end);

T.subject = join(parts(:,1:end-1), '_');
T.subject = string(T.subject);

% cohort = C or P (from subject string)
tokens = regexp(T.subject, 'CAL_(C\d+|P\d+)', 'tokens');

T.cohort = string(cellfun(@(x) x{1}, tokens, 'UniformOutput', false));

% --- ADD THIS HERE ---
% groupType = extractBetween(T.cohort, 1, 1);
% groupType = string(groupType);

% T.cohort = string(extractBetween(T.subject, "CAL_", "_"));

% groupType = extractBefore(T.cohort, 2);
% groupType = string(groupType);

%% DISTRIBUTIONAL ANALYSIS
%% Get features
% features = {
%     'total_volume'
%     'num_branches'
%     'bifurcation_count'
%     'endpoint_count'
%     'num_components'
%     'largest_component_fraction'
%     'largest_component_volume_mm3'
%     'mean_radius'
%     'mean_tortuosity'
%     'total_branch_length'
% };


subjects = unique(T.subject);
visits = ["V1","V2","V3"];

%% spearman preprocessing
data = struct();

features = ["total_volume","num_branches","bifurcation_count", ...
            "endpoint_count","num_components", ...
            "largest_component_fraction","mean_radius", ...
            "mean_tortuosity","total_branch_length"];

for f = 1:length(features)
    feat = features(f);
    
    M = nan(length(subjects), length(visits));
    
    for i = 1:length(subjects)
        for j = 1:length(visits)
            
            idx = T.subject == subjects(i) & T.visit == visits(j);
            
            if any(idx)
                M(i,j) = T{idx, feat};
            end
        end
    end
    
    data.(feat) = M;
end
%% spearman rank stability
spearman_results = table();

row = 1;

for f = 1:length(features)
    feat = features(f);
    M = data.(feat);
    
    v1 = M(:,1);
    v2 = M(:,2);
    v3 = M(:,3);
    
    spearman_results.feature(row) = feat;
    spearman_results.V1_V2(row) = corr(v1, v2, 'Type','Spearman','Rows','complete');
    spearman_results.V1_V3(row) = corr(v1, v3, 'Type','Spearman','Rows','complete');
    spearman_results.V2_V3(row) = corr(v2, v3, 'Type','Spearman','Rows','complete');
    
    row = row + 1;
end

disp(spearman_results)
%% heatmap
figure;

dataMat = [
    spearman_results.V1_V2, ...
    spearman_results.V1_V3, ...
    spearman_results.V2_V3
];

xLabels = ["V1-V2","V1-V3","V2-V3"]';   % FORCE column vector
yLabels = string(spearman_results.feature);

figure;

heatmap(xLabels, yLabels, dataMat, 'Interpreter','none');
title("Spearman Rank Stability Across Visits");
%% Plot hists
for i=1:length(features)
    figure('IntegerHandle','off', ...
       'Name', " Hist " + features{i}, ...
       'NumberTitle','off');

    histogram(T.(features{i}), 10);

    xlabel(features{i}, 'Interpreter','none')
    ylabel('Count')

    title(features{i}, 'Interpreter','none')
    grid minor
end

%% boxplots
for i=1:length(features)
    figure('IntegerHandle','off', ...
       'Name', "Box " + features{i}, ...
       'NumberTitle','off');

    boxplot(T.(features{i}))
    xticklabels({})
    xlabel(features{i}, 'Interpreter','none')

    title(['Boxplot: ' features{i}], 'Interpreter','none')
    grid minor
end

%% Summary statistics
summary_table = table();

for i=1:length(features)
    x = T.(features{i});
    summary_table.feature{i}=features{i};
    summary_table.mean(i)=mean(x);
    summary_table.std(i)=std(x);
    summary_table.median(i)=median(x);
    summary_table.min(i)=min(x);
    summary_table.max(i)=max(x);

end
disp(summary_table)


%% LONGITUDINAL CONSISTENCY (for every subject)

metrics = {
    'total_volume'
    'num_branches'
    'largest_component_fraction'
    'mean_radius'
    'mean_tortuosity'
    'total_branch_length'
};

for m=1:length(metrics)
    figure('IntegerHandle','off', ...
           'Name', "Longitude " + metrics{m}, ...
           'NumberTitle','off')
    hold on
    
    for i=1:length(subjects)
        idx=T.subject==subjects(i);
        temp=T(idx,:);
        
        [~,ord]=sort(temp.visit);
        temp=temp(ord,:);
    
        plot(1:height(temp), temp.(metrics{m}),'-o');
    end
    xticks([1 2 3])
    xticklabels({'V1','V2','V3'})
    ylabel(metrics{m}, 'Interpreter','none')
    
    title(metrics{m},...
    'Interpreter','none')
    legend(subjects, 'Interpreter','none')
    grid minor
    
    hold off

end

%% split by patients vs controls

figure
hold on

colors = containers.Map;
colors("C") = [0 0.447 0.741];
colors("P") = [0.85 0.325 0.098];

for i = 1:length(subjects)

    idx = T.subject == subjects(i);
    temp = T(idx,:);

    [~,ord] = sort(temp.visit);
    temp = temp(ord,:);

    cohort = temp.cohort(1);
    groupType = extractBefore(cohort, 2);

    plot(1:height(temp), temp.total_volume, '-o', ...
        'Color', colors(groupType), ...
        'HandleVisibility','off');

end

xticks([1 2 3])
xticklabels({'V1','V2','V3'})
xlabel('Visit')
ylabel('Total Volume')
title('Longitudinal Total Volume (C vs P)')

% add dummy legend entries
plot(nan,nan,'-o','Color',colors("C"),'DisplayName','Controls (C)')
plot(nan,nan,'-o','Color',colors("P"),'DisplayName','Patients (P)')

legend
hold off

%% within subject coefficient of variation
% Low CVw (e.g., <5–10%): Measurements are highly repeatable within individuals.
% Moderate CVw (e.g., 10–20%): There is noticeable biological or measurement variability.
% High CVw (>20%): Measurements fluctuate considerably within individuals.

CV_table = table();

for s=1:length(subjects)
    idx=T.subject==subjects(s);
    temp=T(idx,:);
    CV_table.subject(s)=subjects(s);
    
    for j=1:length(metrics)
        x=temp.(metrics{j});
        cv = std(x)/mean(x);
        CV_table.(metrics{j})(s)=cv;
    end

end
disp(CV_table)