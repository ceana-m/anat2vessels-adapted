import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Mean Tortuosity: Control vs High UMN Burden ALS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_combat_site_distributions(df_original, df_combat):
    """
    Plot site-wise distributions of selected vascular features
    before and after ComBat harmonization.

    Features:
        - total_volume
        - num_branches
        - mean_tortuosity
    """

    features = [
        "total_volume",
        "num_branches",
        "max_tortuosity"
    ]

    feature_labels = {
        "total_volume": "Total vessel volume (mm³)",
        "num_branches": "Number of branches",
        "max_tortuosity": "Maximum vessel tortuosity"
    }

    # ------------------------------------------------------------
    # Check required columns
    # ------------------------------------------------------------

    required_columns = [
        "Site",
        "Cohort"
    ] + features

    for df_name, df in [
        ("df_original", df_original),
        ("df_combat", df_combat)
    ]:
        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{df_name} is missing columns: {missing}"
            )

    # ------------------------------------------------------------
    # Use the same site order for every panel
    # ------------------------------------------------------------

    site_order = [
        "Calgary",
        "Edmonton",
        "Miami",
        "Montreal",
        "Quebec",
        "Toronto",
        "Utah"
    ]

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(12, 12)
    )

    # ------------------------------------------------------------
    # Plot each feature
    # ------------------------------------------------------------

    for row, feature in enumerate(features):

        # ========================================================
        # Raw data
        # ========================================================

        ax_raw = axes[row, 0]

        raw_data = []
        raw_positions = []

        for i, site in enumerate(site_order, start=1):

            values = pd.to_numeric(
                df_original.loc[
                    df_original["Site"] == site,
                    feature
                ],
                errors="coerce"
            ).dropna()

            raw_data.append(values.values)
            raw_positions.append(i)

        ax_raw.boxplot(
            raw_data,
            positions=raw_positions,
            widths=0.55,
            showfliers=False
        )

        # Add individual observations
        rng = np.random.default_rng(42)

        for i, values in enumerate(raw_data, start=1):

            if len(values) == 0:
                continue

            jitter = rng.uniform(
                -0.15,
                0.15,
                size=len(values)
            )

            ax_raw.scatter(
                np.full(len(values), i) + jitter,
                values,
                s=12,
                alpha=0.25
            )

        ax_raw.set_title("Before ComBat")
        ax_raw.set_ylabel(feature_labels[feature])

        ax_raw.set_xticks(range(1, len(site_order) + 1))
        ax_raw.set_xticklabels(
            site_order,
            rotation=45,
            ha="right"
        )

        # ========================================================
        # ComBat data
        # ========================================================

        ax_combat = axes[row, 1]

        combat_data = []
        combat_positions = []

        for i, site in enumerate(site_order, start=1):

            values = pd.to_numeric(
                df_combat.loc[
                    df_combat["Site"] == site,
                    feature
                ],
                errors="coerce"
            ).dropna()

            combat_data.append(values.values)
            combat_positions.append(i)

        ax_combat.boxplot(
            combat_data,
            positions=combat_positions,
            widths=0.55,
            showfliers=False
        )

        # Add individual observations
        rng = np.random.default_rng(42)

        for i, values in enumerate(combat_data, start=1):

            if len(values) == 0:
                continue

            jitter = rng.uniform(
                -0.15,
                0.15,
                size=len(values)
            )

            ax_combat.scatter(
                np.full(len(values), i) + jitter,
                values,
                s=12,
                alpha=0.25
            )

        ax_combat.set_title("After ComBat")

        ax_combat.set_xticks(
            range(1, len(site_order) + 1)
        )

        ax_combat.set_xticklabels(
            site_order,
            rotation=45,
            ha="right"
        )

        # --------------------------------------------------------
        # Use identical y-axis limits within each feature
        # --------------------------------------------------------

        all_values = np.concatenate(
            [raw_data[i] for i in range(len(raw_data))]
            + [combat_data[i] for i in range(len(combat_data))]
        )

        if len(all_values) > 0:

            y_min = np.min(all_values)
            y_max = np.max(all_values)

            y_range = y_max - y_min

            if y_range == 0:
                y_range = 1

            ax_raw.set_ylim(
                y_min - 0.05 * y_range,
                y_max + 0.05 * y_range
            )

            ax_combat.set_ylim(
                y_min - 0.05 * y_range,
                y_max + 0.05 * y_range
            )

        # --------------------------------------------------------
        # Clean appearance
        # --------------------------------------------------------

        for ax in [ax_raw, ax_combat]:

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------
    # Column labels
    # ------------------------------------------------------------

    axes[0, 0].set_title("Before ComBat")
    axes[0, 1].set_title("After ComBat")

    # ------------------------------------------------------------
    # Overall layout
    # ------------------------------------------------------------

    plt.tight_layout(h_pad=3.0)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    plt.savefig(
        "combat_site_distributions.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_combat_site_effects():

    features = [
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
        "max_branch_length"
    ]

    raw_f = [
        62.493979,
        78.221131,
        78.760860,
        17.729303,
        14.424667,
        23.068290,
        6.875641,
        4.623471,
        94.937000,
        24.304526,
        6.576312
    ]

    combat_f = [
        0.110080,
        0.315020,
        0.185896,
        0.074776,
        1.169472,
        0.447762,
        0.089054,
        0.114404,
        0.130341,
        0.176435,
        0.210958
    ]

    # Reverse order so the first feature appears at the top
    features = features[::-1]
    raw_f = raw_f[::-1]
    combat_f = combat_f[::-1]

    y = np.arange(len(features))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 7),
        sharey=True
    )

    # ------------------------------------------------------------
    # Before ComBat
    # ------------------------------------------------------------

    axes[0].barh(
        y,
        raw_f
    )

    axes[0].set_title("Before ComBat")
    axes[0].set_xlabel("Site ANOVA F-statistic")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(features)

    # ------------------------------------------------------------
    # After ComBat
    # ------------------------------------------------------------

    axes[1].barh(
        y,
        combat_f
    )

    axes[1].set_title("After ComBat")
    axes[1].set_xlabel("Site ANOVA F-statistic")

    # IMPORTANT:
    # Use the same x-axis scale for both panels
    axes[0].set_xlim(0, 100)
    axes[1].set_xlim(0, 100)

    # ------------------------------------------------------------
    # Clean appearance
    # ------------------------------------------------------------

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Reduction of site-associated variation following ComBat harmonization"
    )

    plt.tight_layout()

    plt.savefig(
        "combat_site_effects.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_mean_tortuosity_high_umn_vs_control(df_combat_v1):

    umn_col = "UMNBurden_w_PseudobulbarScore"
    feature = "mean_tortuosity"

    # # ------------------------------------------------------------
    # # 0. Check that the necessary columns exist
    # # ------------------------------------------------------------

    # required_columns = [umn_col, feature, "Cohort"]

    # for col in required_columns:
    #     if col not in df_combat_v1.columns:
    #         raise ValueError(f"Column '{col}' not found in DataFrame.")
    # ------------------------------------------------------------
    # 1. Separate controls and ALS patients
    # ------------------------------------------------------------

    controls = df_combat_v1[
        df_combat_v1["Cohort"] == "Control"
    ].copy()

    patients = df_combat_v1[
        df_combat_v1["Cohort"] == "Patient"
    ].copy()

    # ------------------------------------------------------------
    # 2. Calculate median UMN burden among ALS patients
    # ------------------------------------------------------------

    umn_median = patients[umn_col].median()

    print(f"UMN burden median: {umn_median}")

    # ------------------------------------------------------------
    # 3. Select high-UMN ALS patients
    # ------------------------------------------------------------

    high_umn = patients[
        patients[umn_col] > umn_median
    ].copy()

    # ------------------------------------------------------------
    # 4. Remove missing values for the plotted feature
    # ------------------------------------------------------------

    controls_plot = controls[feature].dropna()
    high_umn_plot = high_umn[feature].dropna()

    print("\n=== Plot Group Sizes ===")
    print(f"Control: {len(controls_plot)}")
    print(f"High UMN ALS: {len(high_umn_plot)}")

    # ------------------------------------------------------------
    # 5. Summary statistics
    # ------------------------------------------------------------

    print("\n=== Summary Statistics ===")

    print("\nControl:")
    print(controls_plot.describe())

    print("\nHigh UMN ALS:")
    print(high_umn_plot.describe())

    # ------------------------------------------------------------
    # 6. Create box plot
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 5))

    groups = [
        controls_plot.values,
        high_umn_plot.values
    ]

    ax.boxplot(
        groups,
        positions=[1, 2],
        widths=0.45,
        showfliers=False
    )

    # ------------------------------------------------------------
    # 7. Add individual participant points
    # ------------------------------------------------------------

    rng = np.random.default_rng(42)

    for x, values in zip([1, 2], groups):

        jitter = rng.uniform(
            -0.12,
            0.12,
            size=len(values)
        )

        ax.scatter(
            np.full(len(values), x) + jitter,
            values,
            alpha=0.30,
            s=18
        )

    # ------------------------------------------------------------
    # 8. Labels
    # ------------------------------------------------------------

    ax.set_xticks([1, 2])

    ax.set_xticklabels([
        "Control",
        "High UMN ALS"
    ])

    ax.set_ylabel("Mean vessel tortuosity")

    # ------------------------------------------------------------
    # 9. Add statistical annotation
    # ------------------------------------------------------------

    # Your existing statistical result
    # fdr_p = 1 #0.03167

    y_max = max(
        controls_plot.max(),
        high_umn_plot.max()
    )

    y_min = min(
        controls_plot.min(),
        high_umn_plot.min()
    )

    y_range = y_max - y_min

    # Position of significance bracket
    # bracket_y = y_max + 0.08 * y_range
    # text_y = y_max + 0.11 * y_range

    # Bracket
    # ax.plot(
    #     [1, 1, 2, 2],
    #     [
    #         bracket_y,
    #         bracket_y + 0.02 * y_range,
    #         bracket_y + 0.02 * y_range,
    #         bracket_y
    #     ],
    #     linewidth=1
    # )

    # ax.text(
    #     1.5,
    #     text_y,
    #     f"FDR-adjusted p = {fdr_p:.4f}",
    #     ha="center",
    #     va="bottom"
    # )

    # ------------------------------------------------------------
    # 10. Clean up appearance
    # ------------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ax.set_title(
    #     "Mean vessel tortuosity by UMN burden"
    # )

    plt.tight_layout()

    # ------------------------------------------------------------
    # 11. Save high-resolution figure
    # ------------------------------------------------------------

    plt.savefig(
        "mean_tortuosity_high_UMN_vs_control.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()