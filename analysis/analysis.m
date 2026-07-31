clear all; close all;
%% Load table
featureTable = readtable('features.csv');

%% Extract subject and visit
parts = split(string(featureTable.sub_id), '_');

featureTable.visit = parts(:,end);

featureTable.subject = join(parts(:,1:end-1), '_');
featureTable.subject = string(featureTable.subject);

% cohort = C or P (from subject string)
featureTable.groupType = strings(height(featureTable),1);
for i = 1:height(featureTable)

    match = regexp(featureTable.subject(i), '_(C|P)\d+$', 'tokens');

    if ~isempty(match)
        featureTable.groupType(i) = string(match{1}{1});
    else
        warning("Could not determine cohort for %s", featureTable.subject(i))
    end

end
featureTable.groupType = categorical(featureTable.groupType, ["C","P"]);

subjects = unique(featureTable.subject);

%% Define constants
VISITS = ["V1","V2","V3"];
FEATURE_NAMES = [
    "num_branches", ...
    "total_volume", ...
    "bifurcations", ...
    "endpoints", ...
    "mean_radius", ...
    "max_radius", ...
    "mean_tortuosity", ...
    "max_tortuosity", ...
    "total_branch_length", ...
    "mean_branch_length", ...
    "max_branch_length", ...
];

NUM_BINS_HIST = 10;
BOXPLOT_GROUPING = "cohortvisit"; % "cohortvisit", "cohort", "visit"
LONGITUDINAL_GROUPING = 'cohort'; % "all","cohort","control","patient"

%% DISTRIBUTIONAL ANALYSIS
plotBoxplots(featureTable, FEATURE_NAMES, BOXPLOT_GROUPING)
plotHistograms(featureTable, FEATURE_NAMES, NUM_BINS_HIST)
printSummaryTable(featureTable, FEATURE_NAMES)

%% LONGITUDINAL ANALYSIS
plotMetricsOverTime(featureTable, FEATURE_NAMES, subjects, VISITS, LONGITUDINAL_GROUPING)

spearmanProcessing(featureTable, FEATURE_NAMES, subjects, VISITS)

%% Helper functions
function plotMetricsOverTime(featureTable, featureNames, subjects, visits, groupMode)
    % argument validation
    if nargin < 5
        groupMode = "all";
    end
    
    groupMode = lower(string(groupMode));
    
    colors = containers.Map;
    colors("C") = [0 0.4470 0.7410];
    colors("P") = [0.8500 0.3250 0.0980];
    
    validModes = ["all","cohort","control","patient"];
    if ~ismember(groupMode, validModes)
        error("Invalid groupMode. Use: all, cohort, control, patient");
    end
    
    % process features
    for m = 1:length(featureNames)
        figure('IntegerHandle','off', ...
               'Name',"Longitudinal " + featureNames{m}, ...
               'NumberTitle','off')
        hold on

        %% =========================
        %  INDIVIDUAL SUBJECTS
        %% =========================
        for i = 1:length(subjects)
            idx = featureTable.subject == subjects(i);
            temp = featureTable(idx,:);
    
            [~,ord] = sort(temp.visit);
            temp = temp(ord,:);
    
            groupType = string(temp.groupType(1));  % "C" or "P"
    
            switch groupMode
                case "control"
                    if groupType ~= "C", continue; end
                case "patient"
                    if groupType ~= "P", continue; end
            end
    
            % plot individuals
            if groupMode == "cohort"
                plot(1:height(temp), temp.(featureNames{m}), '-o', ...
                    'Color', colors(groupType), ...
                    'DisplayName', char(subjects(i)));
            else
                plot(1:height(temp), temp.(featureNames{m}), '-o', ...
                    'DisplayName', char(subjects(i)));
            end
        end
    
        %% =========================
        %  GROUP MEANS
        %% =========================
    
        if groupMode == "cohort" || groupMode == "control" || groupMode == "patient"
    
            meanControl = nan(1,length(visits));
            meanPatient = nan(1,length(visits));
    
            for v = 1:length(visits)
    
                idxV = featureTable.visit == visits(v);
    
                if groupMode ~= "patient"
                    idxC = idxV & featureTable.groupType == "C";
                    meanControl(v) = mean(featureTable{idxC, featureNames{m}}, 'omitnan');
                end
    
                if groupMode ~= "control"
                    idxP = idxV & featureTable.groupType == "P";
                    meanPatient(v) = mean(featureTable{idxP, featureNames{m}}, 'omitnan');
                end
    
            end
    
            % plot control mean
            if groupMode ~= "patient"
                plot(1:length(visits), meanControl, '--s', ...
                    'LineWidth', 3, 'Color', colors("C"), ...
                    'DisplayName', 'Control Mean');
            end
    
            % plot patient mean
            if groupMode ~= "control"
                plot(1:length(visits), meanPatient, '--s', ...
                    'LineWidth', 3, 'Color', colors("P"), ...
                    'DisplayName', 'Patient Mean');
            end
    
        end
    
        %% =========================
        %  FORMATTING
        %% =========================
        xticks(1:length(visits))
        xticklabels(visits)
    
        xlabel('Visit')
        ylabel(featureNames{m}, 'Interpreter','none')
        title(featureNames{m}, 'Interpreter','none')
    
        grid minor
    
        legend('Location','best', 'Interpreter','none')
    
        hold off
    end
end

function printSummaryTable(featureTable, featureNames)
    summary_table = table();

    for i=1:length(featureNames)
        x = featureTable.(featureNames{i});
        summary_table.feature{i}=featureNames{i};
        summary_table.mean(i)=mean(x);
        summary_table.std(i)=std(x);
        summary_table.median(i)=median(x);
        summary_table.min(i)=min(x);
        summary_table.max(i)=max(x);
    
    end
    disp(summary_table)
end

function plotBoxplots(featureTable, featureNames, grouping)

    [controls, patients] = getNumControlsPatients(featureTable);
    
    for i=1:length(featureNames)
        boxTitle = "Boxplot " + featureNames{i};
        figure('IntegerHandle','off', ...
           'Name', boxTitle, ...
           'NumberTitle','off');
        if strcmp(grouping, "visit")
            % just visits
            boxplot(featureTable.(featureNames{i}), T.visit)
        elseif strcmp(grouping, "cohort")
            % just patients
            boxplot(featureTable.(featureNames{i}),T.groupType)
        elseif strcmp(grouping, "cohortvisit")
            % visits x patients
            groups = strcat(string(featureTable.groupType),"_",string(featureTable.visit));
            boxplot(featureTable.(featureNames{i}),groups)
        else
            error("Incorrect grouping. Must be visit, cohort, or cohortvisit")
        end
    
        xlabel('Visit')
        ylabel(featureNames{i},'Interpreter','none')
        
        title(addCountsToTitle(boxTitle, controls, patients),'Interpreter','none')
        
        grid minor
    end
end

function [controls, patients] = getNumControlsPatients(featureTable)
    [~, idx] = unique(featureTable.subject);

    controls = sum(featureTable.groupType(idx) == "C");
    patients = sum(featureTable.groupType(idx) == "P");
end

function updatedTitle = addCountsToTitle(title, controls, patients)
     updatedTitle = title + " (C = " + controls + ", P = " + patients + ")";
end

function plotHistograms(featureTable, featureNames, numBins)
    for i=1:length(featureNames)
        [controls, patients] = getNumControlsPatients(featureTable);
        histTitle = "Hist " + featureNames{i};
        figure('IntegerHandle','off', ...
           'Name', histTitle, ...
           'NumberTitle','off');
    
        histogram(featureTable.(featureNames{i}), numBins);
    
        xlabel(featureNames{i}, 'Interpreter','none')
        ylabel('Count')
    
        title(addCountsToTitle(featureNames{i}, controls, patients), 'Interpreter','none')
        grid minor
    end
end

function spearmanProcessing(featureTable, featureNames, subjects, visits)
    % prepare data
    data = struct();
    
    for f = 1:length(featureNames)
        feat = featureNames(f);
        
        metricsAcrossVisits = nan(length(subjects), length(visits));
        
        for i = 1:length(subjects)
            for j = 1:length(visits)
                
                idx = featureTable.subject == subjects(i) & featureTable.visit == visits(j);
                
                if any(idx)
                    metricsAcrossVisits(i,j) = featureTable{idx, feat};
                end
            end
        end
        
        data.(feat) = metricsAcrossVisits;
    end

    % create spearman rank stability results
    spearman_results = table();
    row = 1;
    for f = 1:length(featureNames)
        feat = featureNames(f);
        metricsAcrossVisits = data.(feat);
        
        v1 = metricsAcrossVisits(:,1);
        v2 = metricsAcrossVisits(:,2);
        v3 = metricsAcrossVisits(:,3);
        
        spearman_results.feature(row) = feat;
        spearman_results.V1_V2(row) = corr(v1, v2, 'Type','Spearman','Rows','complete');
        spearman_results.V1_V3(row) = corr(v1, v3, 'Type','Spearman','Rows','complete');
        spearman_results.V2_V3(row) = corr(v2, v3, 'Type','Spearman','Rows','complete');
        
        row = row + 1;
    end
    disp(spearman_results)

    % plot heatmap
    dataMat = [
        spearman_results.V1_V2, ...
        spearman_results.V1_V3, ...
        spearman_results.V2_V3
    ];
    
    xLabels = ["V1-V2","V1-V3","V2-V3"]';
    yLabels = string(spearman_results.feature);
    
    figure;
    
    heatmap(xLabels, yLabels, dataMat, 'Interpreter','none');
    title("Spearman Rank Stability Across Visits");
end
