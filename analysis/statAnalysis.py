import os

# Limit BLAS/OpenMP threading before importing numerical libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from scipy.stats import ttest_ind, spearmanr

from neuroCombat import neuroCombat

from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

import pingouin as pg

import manuscriptFigs as mf


# ============================================================
# Configuration
# ============================================================

FEATURES_PATH = (
    r"Q:\anat2vessels\features\features.csv"
)

METADATA_PATH = (
    r"Q:\Notes+Files\job\S26 WorkLearn\Data"
    r"\Final_Data_sheet_April2025.xlsx"
)

FEATURE_COLUMNS = [
    "num_branches",
    "total_volume",
    "bifurcations",
    "endpoints",
    "mean_radius",
    "max_radius",
    "mean_tortuosity",
    "max_tortuosity",
    "total_branch_length",
    "mean_branch_length",
    "max_branch_length",
]

GROUP_COLUMN = "Cohort"
SUBJECT_COLUMN = "Filename"
VISIT_COLUMN = "Visit Label"
TIME_COLUMN = "RelativeTimeYears"

USE_COMBAT = True
RANDOM_SLOPE = True
INCLUDE_SITE_MIXED_EFFECTS = False


# ============================================================
# Data loading and preparation
# ============================================================

def load_data():
    """Load feature and metadata files."""

    features = pd.read_csv(FEATURES_PATH)
    metadata = pd.read_excel(METADATA_PATH)

    return features, metadata


def prepare_metadata(metadata):
    """Standardize metadata and create subject/visit identifiers."""

    metadata = metadata.copy()

    visit_map = {
        "Visit 1": "V1",
        "Visit 2": "V2",
        "Visit 3": "V3",
    }

    metadata["Visit Label"] = (
        metadata["Visit Label"]
        .astype(str)
        .str.strip()
    )

    metadata["visit"] = metadata["Visit Label"].map(visit_map)

    metadata["sub_id"] = (
        metadata["Filename"]
        + "_"
        + metadata["visit"]
    )

    # Carry baseline demographic information across visits
    metadata["Age"] = (
        metadata
        .groupby("Filename")["Age"]
        .transform("first")
    )

    metadata["Sex"] = (
        metadata
        .groupby("Filename")["Sex"]
        .transform("first")
    )

    return metadata


def standardize_df(df):
    """Standardize column names and variable formats."""

    df = df.copy()

    # Standardize cohort name
    df = df.rename(
        columns={
            "Patient or Control": "Cohort"
        }
    )

    # Convert visit labels to numeric visit number
    if "Visit Label" in df.columns:

        visit_map = {
            "Visit 1": 1,
            "Visit 2": 2,
            "Visit 3": 3,
        }

        df["Visit"] = (
            df["Visit Label"]
            .map(visit_map)
        )

    # Convert MRI date
    if "MRI_Date" in df.columns:

        df["MRI_Date"] = pd.to_datetime(
            df["MRI_Date"],
            errors="coerce"
        )

    # Extract site from filename if necessary
    if "Site" not in df.columns:

        df["Site"] = (
            df["Filename"]
            .str.split("_")
            .str[1]
        )

    df["Site"] = (
        df["Site"]
        .astype(str)
        .str.strip()
    )

    # Standardize categorical variables
    for col in ["Sex", "Cohort", "Site"]:

        if col in df.columns:
            df[col] = df[col].astype(str)

    # Convert clinical variable
    if "UMNBurden_w_PseudobulbarScore" in df.columns:

        df["UMNBurden_w_PseudobulbarScore"] = (
            pd.to_numeric(
                df["UMNBurden_w_PseudobulbarScore"],
                errors="coerce"
            )
        )

    return df


def validate_dataframe(df):
    """Check required columns and basic data structure."""

    required = [
        "Filename",
        "Cohort",
        "Sex",
        "Age",
        "Site",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("\n=== DATA VALIDATION ===")
    print(f"Observations: {len(df)}")
    print(f"Participants: {df['Filename'].nunique()}")

    print("\nCohort:")
    print(df["Cohort"].value_counts())

    print("\nSite:")
    print(df["Site"].value_counts())


def merge_data(features, metadata):
    """Merge feature data with metadata using sub_id."""

    matches = features["sub_id"].isin(
        metadata["sub_id"]
    )

    print(
        f"Matched {matches.sum()} "
        f"of {len(features)} feature rows"
    )

    df = features.merge(
        metadata,
        on="sub_id",
        how="inner",
    )

    return df


def get_visit_data(
    df,
    visit="V1",
    controls_only=False
):
    """Return data for a specific visit."""

    result = df[
        df["visit"] == visit
    ].copy()

    if controls_only:

        result = result[
            result["Cohort"] == "Control"
        ]

    return result


def prepare_longitudinal_data(
    df,
    visit_col="Visit Label",
    in_months=False,
    relative=True
):
    """Calculate time relative to each participant's baseline MRI."""

    if visit_col not in df.columns:
        raise ValueError(
            f"{visit_col} not found in dataframe columns"
        )

    df = df.copy()

    df["MRI_Date"] = pd.to_datetime(
        df["MRI_Date"],
        errors="coerce"
    )

    df["DaysFromBaseline"] = (
        df["MRI_Date"]
        - df.groupby("Filename")["MRI_Date"]
        .transform("min")
    ).dt.days

    if in_months:

        if relative:
            raise ValueError(
                "Relative time in months has not "
                "been implemented and tested."
            )

        visit_to_months = {
            "Visit 1": 0,
            "Visit 2": 4,
            "Visit 3": 8,
        }

        df["TimeMonths"] = (
            df[visit_col]
            .map(visit_to_months)
        )

    else:

        if relative:

            df["RelativeTimeYears"] = (
                df["DaysFromBaseline"] / 365.25
            )

        else:

            visit_to_years = {
                "Visit 1": 0.0,
                "Visit 2": 4 / 12,
                "Visit 3": 8 / 12,
            }

            df["TimeYears"] = (
                df[visit_col]
                .map(visit_to_years)
            )

    return df


# ============================================================
# Baseline statistical analysis
# ============================================================

def welch_test(
    df,
    feature_columns,
    group_col,
    comparison_group,
    control_group="Control",
):
    """Compare two groups using Welch's t-test."""

    results = []

    for feature in feature_columns:

        control_data = (
            df[
                df[group_col] == control_group
            ][feature]
            .dropna()
        )

        comparison_data = (
            df[
                df[group_col] == comparison_group
            ][feature]
            .dropna()
        )

        t, p = ttest_ind(
            comparison_data,
            control_data,
            equal_var=False
        )

        results.append({
            "Feature": feature,

            "Control n": len(control_data),
            "Control mean": control_data.mean(),
            "Control SD": control_data.std(),

            f"{comparison_group} n":
                len(comparison_data),

            f"{comparison_group} mean":
                comparison_data.mean(),

            f"{comparison_group} SD":
                comparison_data.std(),

            "Mean difference":
                comparison_data.mean()
                - control_data.mean(),

            "t stat": t,
            "p value": p,
        })

    results_df = pd.DataFrame(results)

    results_df["FDR adjusted p"] = (
        multipletests(
            results_df["p value"],
            method="fdr_bh"
        )[1]
    )

    return results_df


def compute_effect_sizes(
    df,
    feature_columns,
    group_col
):
    """Calculate Hedges' g for Patient vs Control."""

    results = []

    for feature in feature_columns:

        controls = (
            df[
                df[group_col] == "Control"
            ][feature]
            .dropna()
        )

        patients = (
            df[
                df[group_col] == "Patient"
            ][feature]
            .dropna()
        )

        effect = pg.compute_effsize(
            patients,
            controls,
            eftype="hedges"
        )

        results.append({
            "Feature": feature,
            "Hedges g": effect
        })

    return pd.DataFrame(results)


# ============================================================
# Clinical subgroup analysis
# ============================================================

def assign_median_groups(
    df,
    split_column,
    group_col="analysis_group",
    control_label="Control",
    patient_label="Patient"
):
    """Assign Low/High patient groups using the patient median."""

    df = df.copy()

    df[split_column] = pd.to_numeric(
        df[split_column]
        .astype(str)
        .str.strip(),
        errors="coerce"
    )

    patient_mask = (
        (df["Cohort"] == patient_label)
        & df[split_column].notna()
    )

    median = (
        df.loc[
            patient_mask,
            split_column
        ].median()
    )

    df[group_col] = pd.Series(
        index=df.index,
        dtype="object"
    )

    # Controls
    control_mask = (
        df["Cohort"] == control_label
    )

    df.loc[
        control_mask,
        group_col
    ] = "Control"

    # Low patients
    df.loc[
        patient_mask
        & (df[split_column] <= median),
        group_col
    ] = "Low"

    # High patients
    df.loc[
        patient_mask
        & (df[split_column] > median),
        group_col
    ] = "High"

    print(
        f"{split_column} median = "
        f"{median:.3f}"
    )

    print(
        f"Patients equal to median: "
        f"{(df.loc[patient_mask, split_column] == median).sum()}"
    )

    print(
        "\nAssigned groups:"
    )

    print(
        df[group_col]
        .value_counts(dropna=False)
    )

    return df


def run_median_subgroup_analysis(
    df,
    split_column,
    feature_columns,
    group_col="analysis_group"
):
    """Compare Low and High clinical-burden groups with controls."""

    df_grouped = assign_median_groups(
        df,
        split_column,
        group_col=group_col
    )

    low_results = welch_test(
        df_grouped,
        feature_columns,
        group_col=group_col,
        comparison_group="Low"
    )

    high_results = welch_test(
        df_grouped,
        feature_columns,
        group_col=group_col,
        comparison_group="High"
    )

    return {
        "data": df_grouped,
        "low": low_results,
        "high": high_results,
    }


def clinical_correlation(
    df,
    feature_columns,
    clinical_column
):
    """Calculate Spearman correlations between features and clinical variables."""

    results = []

    clinical_series = pd.to_numeric(
        df[clinical_column],
        errors="coerce"
    )

    for feature in feature_columns:

        feature_series = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        valid_mask = (
            feature_series.notna()
            & clinical_series.notna()
        )

        x = feature_series[valid_mask]
        y = clinical_series[valid_mask]

        n_samples = len(x)

        if (
            n_samples < 2
            or x.nunique() <= 1
            or y.nunique() <= 1
        ):

            results.append({
                "Feature": feature,
                "Clinical metric": clinical_column,
                "N": n_samples,
                "Spearman rho": np.nan,
                "p value": np.nan,
            })

            continue

        rho, p = spearmanr(x, y)

        results.append({
            "Feature": feature,
            "Clinical metric": clinical_column,
            "N": n_samples,
            "Spearman rho": rho,
            "p value": p,
        })

    results_df = pd.DataFrame(results)

    results_df["FDR"] = np.nan

    valid = results_df["p value"].notna()

    if valid.any():

        results_df.loc[
            valid,
            "FDR"
        ] = multipletests(
            results_df.loc[
                valid,
                "p value"
            ],
            method="fdr_bh"
        )[1]

    return results_df


# ============================================================
# ComBat harmonization
# ============================================================

def run_combat(df):
    """Apply ComBat harmonization across imaging sites."""

    covars = df[
        [
            "Site",
            "Age",
            "Sex",
            "Cohort",
            "RelativeTimeYears"
        ]
    ].copy()

    for col in ["Site", "Sex", "Cohort"]:
        covars[col] = covars[col].astype(str)

    # ComBat requires complete feature/covariate rows
    valid = (
        df[FEATURE_COLUMNS]
        .notna()
        .all(axis=1)
        & covars.notna().all(axis=1)
    )

    df_valid = df.loc[valid].copy()
    covars = covars.loc[valid].copy()

    feature_matrix = (
        df_valid[FEATURE_COLUMNS]
        .to_numpy()
        .T
    )

    print(
        f"\nComBat observations: "
        f"{feature_matrix.shape[1]}"
    )

    print(
        f"ComBat features: "
        f"{feature_matrix.shape[0]}"
    )

    combat_result = neuroCombat(
        dat=feature_matrix,
        covars=covars,
        batch_col="Site",
        categorical_cols=[
            "Sex",
            "Cohort"
        ],
        continuous_cols=[
            "Age",
            "RelativeTimeYears"
        ],
    )

    harmonized = combat_result["data"].T

    df_combat = df.copy()

    df_combat[FEATURE_COLUMNS] = (
        df_combat[FEATURE_COLUMNS]
        .astype(float)
    )

    df_combat.loc[
        valid,
        FEATURE_COLUMNS
    ] = harmonized

    return df_combat


# ============================================================
# Longitudinal mixed-effects analysis
# ============================================================

def mixed_effects_model(
    df,
    feature_columns,
    time_col,
    diagnosis_col,
    subject_col,
    age_col="Age",
    sex_col="Sex",
    site_col="Site",
    include_site=False,
    random_slope=False,
):
    """
    Longitudinal mixed-effects model.

    Fixed effects:
        Time
        Diagnosis
        Time × Diagnosis
        Age
        Sex
        optional Site

    Random effect:
        Subject-specific intercept

    Optional:
        Subject-specific time slope
    """

    fixed_effects = [
        age_col,
        f"C({sex_col})",
    ]

    if include_site:
        fixed_effects.append(
            f"C({site_col})"
        )

    results = []

    for feature in feature_columns:

        print(
            f"\nRunning mixed model: {feature}"
        )

        formula = (
            f"{feature} ~ "
            + " + ".join(fixed_effects)
            + f" + {time_col}"
            + f" + C({diagnosis_col})"
            + f" + {time_col}:C({diagnosis_col})"
        )

        cols = [
            feature,
            time_col,
            diagnosis_col,
            subject_col,
            age_col,
            sex_col,
            site_col,
        ]

        temp = df[cols].copy()

        temp = temp.dropna()

        temp[diagnosis_col] = (
            temp[diagnosis_col]
            .astype(str)
        )

        temp[sex_col] = (
            temp[sex_col]
            .astype(str)
        )

        temp[site_col] = (
            temp[site_col]
            .astype(str)
        )

        print(
            f"Observations: {len(temp)}, "
            f"Subjects: "
            f"{temp[subject_col].nunique()}"
        )

        try:

            if random_slope:

                model = smf.mixedlm(
                    formula,
                    data=temp,
                    groups=temp[subject_col],
                    re_formula=f"~{time_col}"
                )

            else:

                model = smf.mixedlm(
                    formula,
                    data=temp,
                    groups=temp[subject_col],
                    re_formula="1"
                )

            fit = model.fit(
                method="lbfgs",
                maxiter=1000,
                disp=True
            )

        except Exception as e:

            print(
                f"FAILED: {feature}"
            )

            print(e)

            results.append({
                "Feature": feature,
                "Status": f"Failed: {e}"
            })

            continue

        params = fit.params
        pvalues = fit.pvalues
        conf = fit.conf_int()

        row = {
            "Feature": feature,
            "N observations": len(temp),
            "N subjects":
                temp[subject_col].nunique(),
            "Convergence":
                fit.converged,
        }

        for term in fit.fe_params.index:

            row[f"{term} beta"] = (
                params[term]
            )

            row[f"{term} p"] = (
                pvalues[term]
            )

            row[f"{term} CI lower"] = (
                conf.loc[term, 0]
            )

            row[f"{term} CI upper"] = (
                conf.loc[term, 1]
            )

        interaction_keys = [
            key
            for key in params.index
            if f"{time_col}:C({diagnosis_col})"
            in key
        ]

        if interaction_keys:

            key = interaction_keys[0]

            row["Interaction beta"] = (
                params[key]
            )

            row["Interaction p"] = (
                pvalues[key]
            )

        results.append(row)

    results_df = pd.DataFrame(results)

    # FDR correction for every fixed-effect p-value
    for col in list(results_df.columns):

        if not col.endswith(" p"):
            continue

        valid = results_df[col].notna()

        fdr = pd.Series(
            index=results_df.index,
            dtype=float
        )

        if valid.any():

            fdr.loc[valid] = (
                multipletests(
                    results_df.loc[
                        valid,
                        col
                    ],
                    method="fdr_bh"
                )[1]
            )

        insert_loc = (
            results_df.columns
            .get_loc(col)
            + 1
        )

        results_df.insert(
            loc=insert_loc,
            column=col.replace(
                " p",
                " FDR"
            ),
            value=fdr
        )

    return results_df


# ============================================================
# Result reporting
# ============================================================

def summarize_significant_results(
    results,
    feature_col="Feature",
    p_col="p value",
    fdr_col="FDR",
    label="Analysis",
):
    """Print nominally significant and FDR-significant results."""

    print(
        f"\n=== {label} ==="
    )

    if not all(
        col in results.columns
        for col in [
            feature_col,
            p_col,
            fdr_col
        ]
    ):
        print(
            "Required result columns are missing."
        )
        return

    raw_sig = results[
        results[p_col] < 0.05
    ]

    fdr_sig = results[
        results[fdr_col] < 0.05
    ]

    print("\nRaw p < 0.05:")

    if raw_sig.empty:
        print("None")

    else:

        for _, row in raw_sig.iterrows():

            print(
                f"- {row[feature_col]}: "
                f"p={row[p_col]:.4g}, "
                f"FDR={row[fdr_col]:.4g}"
            )

    print("\nFDR < 0.05:")

    if fdr_sig.empty:
        print("None")

    else:

        for _, row in fdr_sig.iterrows():

            print(
                f"- {row[feature_col]}: "
                f"p={row[p_col]:.4g}, "
                f"FDR={row[fdr_col]:.4g}"
            )


def demographic_summary(
    df,
    group_col="Cohort"
):
    """Print baseline demographic summary."""

    participants = (
        df
        .drop_duplicates("Filename")
        .copy()
    )

    print(
        "\n=== DEMOGRAPHIC SUMMARY ==="
    )

    print("\nParticipants:")
    print(
        participants[group_col]
        .value_counts()
    )

    print(
        "\nTotal participants:",
        len(participants)
    )

    print("\nAge overall:")
    print(
        participants["Age"]
        .describe()
    )

    print("\nAge by cohort:")
    print(
        participants
        .groupby(group_col)["Age"]
        .describe()
    )

    print("\nSex by cohort:")
    print(
        pd.crosstab(
            participants[group_col],
            participants["Sex"]
        )
    )

    print("\nSite by cohort:")
    print(
        pd.crosstab(
            participants["Site"],
            participants[group_col]
        )
    )


# ============================================================
# Analysis wrappers
# ============================================================

def run_baseline_analysis(df_base):

    print(
        "\n================================"
        "\nBASELINE ANALYSIS"
        "\n================================"
    )

    # ALS vs Control
    patient_results = welch_test(
        df_base,
        FEATURE_COLUMNS,
        group_col=GROUP_COLUMN,
        comparison_group="Patient"
    )

    print("\n=== ALS vs Control ===")
    print(patient_results)

    print("\n=== Effect Sizes ===")
    print(
        compute_effect_sizes(
            df_base,
            FEATURE_COLUMNS,
            GROUP_COLUMN
        )
    )

    return patient_results


def run_clinical_subgroup_analysis(df_base):

    print(
        "\n================================"
        "\nCLINICAL SUBGROUP ANALYSIS"
        "\n================================"
    )

    dpr_results = run_median_subgroup_analysis(
        df_base,
        "DiseaseProgressionRate",
        FEATURE_COLUMNS
    )

    umn_without_score = (
        run_median_subgroup_analysis(
            df_base,
            "UMNBurden_w/o_pseudobulbarscore",
            FEATURE_COLUMNS
        )
    )

    umn_with_score = (
        run_median_subgroup_analysis(
            df_base,
            "UMNBurden_w_PseudobulbarScore",
            FEATURE_COLUMNS
        )
    )

    return {
        "dpr": dpr_results,
        "umn_without_score":
            umn_without_score,
        "umn_with_score":
            umn_with_score,
    }


def run_clinical_correlations(df_base):

    print(
        "\n================================"
        "\nCLINICAL CORRELATIONS"
        "\n================================"
    )

    als_baseline = df_base[
        df_base["Cohort"] == "Patient"
    ]

    results = {}

    for clinical_column in [
        "DiseaseProgressionRate",
        "UMNBurden_w/o_pseudobulbarscore",
        "UMNBurden_w_PseudobulbarScore",
    ]:

        result = clinical_correlation(
            als_baseline,
            FEATURE_COLUMNS,
            clinical_column
        )

        results[clinical_column] = result

        print(
            f"\n=== {clinical_column} ==="
        )

        print(result)

    return results


def run_longitudinal_analysis(df_long):

    print(
        "\n================================"
        "\nLONGITUDINAL ANALYSIS"
        "\n================================"
    )

    # Keep only participants with baseline Visit 1
    v1_subjects = (
        df_long.loc[
            df_long["Visit Label"] == "Visit 1",
            "Filename"
        ]
        .unique()
    )

    df_long = df_long[
        df_long["Filename"]
        .isin(v1_subjects)
    ].copy()

    print(
        f"Longitudinal cohort: "
        f"{df_long['Filename'].nunique()} "
        f"participants, "
        f"{len(df_long)} observations"
    )

    mixed_results = mixed_effects_model(
        df=df_long,
        feature_columns=FEATURE_COLUMNS,
        time_col=TIME_COLUMN,
        diagnosis_col="Cohort",
        subject_col=SUBJECT_COLUMN,
        age_col="Age",
        sex_col="Sex",
        site_col="Site",
        include_site=INCLUDE_SITE_MIXED_EFFECTS,
        random_slope=RANDOM_SLOPE,
    )

    return mixed_results


def run_site_anova(df_original, df_analysis):

    from statsmodels.stats.anova import anova_lm

    def site_anova(df):

        results = []

        for feature in FEATURE_COLUMNS:

            temp = df[
                [feature, "Site"]
            ].dropna()

            model = smf.ols(
                f"{feature} ~ C(Site)",
                data=temp
            ).fit()

            anova = anova_lm(model)

            results.append({
                "Feature": feature,
                "F": anova.loc[
                    "C(Site)", "F"
                ],
                "p": anova.loc[
                    "C(Site)", "PR(>F)"
                ],
            })

        return pd.DataFrame(results)

    raw_results = site_anova(
        df_original
    )

    analysis_results = site_anova(
        df_analysis
    )

    print(
        "\n=== Site ANOVA: Raw ==="
    )
    print(raw_results)

    print(
        "\n=== Site ANOVA: Analysis Data ==="
    )
    print(analysis_results)

    return raw_results, analysis_results


# ============================================================
# Manuscript figures
# ============================================================

def generate_manuscript_figures(
    df_original,
    df_analysis,
    df_base
):

    print(
        "\n================================"
        "\nMANUSCRIPT FIGURES"
        "\n================================"
    )

    # Main biological result
    mf.plot_mean_tortuosity_high_umn_vs_control(
        df_base
    )

    # ComBat supplementary figure
    mf.plot_combat_site_distributions(
        df_original,
        df_analysis
    )


# ============================================================
# Main pipeline
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    features, metadata = load_data()

    # --------------------------------------------------------
    # 2. Prepare metadata and merge
    # --------------------------------------------------------

    metadata = prepare_metadata(
        metadata
    )

    df = merge_data(
        features,
        metadata
    )

    df = standardize_df(df)

    validate_dataframe(df)

    # --------------------------------------------------------
    # 3. Prepare longitudinal variables
    # --------------------------------------------------------

    df = prepare_longitudinal_data(
        df,
        visit_col=VISIT_COLUMN,
        in_months=False
    )

    df_original = df.copy()

    # --------------------------------------------------------
    # 4. ComBat harmonization
    # --------------------------------------------------------

    if USE_COMBAT:

        df_analysis = run_combat(
            df_original
        )

    else:

        df_analysis = (
            df_original.copy()
        )

    # --------------------------------------------------------
    # 5. Baseline data
    # --------------------------------------------------------

    df_base = get_visit_data(
        df_analysis,
        visit="V1"
    )

    # --------------------------------------------------------
    # 6. Demographics
    # --------------------------------------------------------

    demographic_summary(
        df_base
    )

    # --------------------------------------------------------
    # 7. Statistical analyses
    # --------------------------------------------------------

    baseline_results = (
        run_baseline_analysis(
            df_base
        )
    )

    subgroup_results = (
        run_clinical_subgroup_analysis(
            df_base
        )
    )

    correlation_results = (
        run_clinical_correlations(
            df_base
        )
    )

    # --------------------------------------------------------
    # 8. Longitudinal analysis
    # --------------------------------------------------------

    mixed_results = (
        run_longitudinal_analysis(
            df_analysis
        )
    )

    # --------------------------------------------------------
    # 9. Site-effect check
    # --------------------------------------------------------

    run_site_anova(
        df_original,
        df_analysis
    )

    # --------------------------------------------------------
    # 10. Report significant results
    # --------------------------------------------------------

    summarize_significant_results(
        baseline_results,
        p_col="p value",
        fdr_col="FDR adjusted p",
        label="Baseline ALS vs Control"
    )

    for label, key in [
        ("Low DPR vs Control",
         ("dpr", "low")),

        ("High DPR vs Control",
         ("dpr", "high")),

        ("Low UMN Burden (without Score) vs Control",
         ("umn_without_score", "low")),

        ("High UMN Burden (without Score) vs Control",
         ("umn_without_score", "high")),

        ("Low UMN Burden (with Score) vs Control",
         ("umn_with_score", "low")),

        ("High UMN Burden (with Score) vs Control",
         ("umn_with_score", "high")),
    ]:

        subgroup_name, result_name = key

        summarize_significant_results(
            subgroup_results[
                subgroup_name
            ][result_name],
            p_col="p value",
            fdr_col="FDR adjusted p",
            label=label
        )

    # --------------------------------------------------------
    # 11. Longitudinal interaction
    # --------------------------------------------------------

    interaction = (
        f"{TIME_COLUMN}:C(Cohort)[T.Patient]"
    )

    if all(
        col in mixed_results.columns
        for col in [
            f"{interaction} beta",
            f"{interaction} p",
            f"{interaction} FDR",
        ]
    ):

        longitudinal_summary = (
            mixed_results[
                [
                    "Feature",
                    f"{interaction} beta",
                    f"{interaction} p",
                    f"{interaction} FDR",
                ]
            ]
            .copy()
        )

        longitudinal_summary = (
            longitudinal_summary.rename(
                columns={
                    f"{interaction} beta":
                        "beta",

                    f"{interaction} p":
                        "p value",

                    f"{interaction} FDR":
                        "FDR",
                }
            )
        )

        summarize_significant_results(
            longitudinal_summary,
            p_col="p value",
            fdr_col="FDR",
            label="Longitudinal Time × Cohort"
        )

    # --------------------------------------------------------
    # 12. Save analysis datasets/results
    # --------------------------------------------------------

    df_original.to_csv(
        "features/features_raw.csv",
        index=False
    )

    df_analysis.to_csv(
        "features/features_combat.csv",
        index=False
    )

    prefix = (
        "combat"
        if USE_COMBAT
        else "raw"
    )

    mixed_results.to_csv(
        f"{prefix}_mixed_effects_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # 13. Manuscript figures
    # --------------------------------------------------------

    generate_manuscript_figures(
        df_original,
        df_analysis,
        df_base
    )

    print(
        "\n================================"
        "\nANALYSIS COMPLETE"
        "\n================================"
    )


if __name__ == "__main__":
    main()