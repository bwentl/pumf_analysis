# %%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn import linear_model
from sklearn import tree
from sklearn.cluster import KMeans

from pumf_2016_data import pumf_2016_util
from pumf_2021_data import pumf_2021_util

# StandardScaler preferred over Normalizer, see note 1
# note 1: https://datascience.stackexchange.com/questions/45900/when-to-use-standard-scaler-and-when-normalizer


# %%

df_2016 = pumf_2016_util.read_indiv_table()
pumf_2016_util.recode_indiv_table(df_2016)
df = pumf_2021_util.read_indiv_table()
pumf_2021_util.recode_indiv_table(df)

# list columns
print(list(df.columns))


# %%

# population of age by household size
# - set fig size
fs = (4, 4)
# - total
plt.figure(figsize=fs)
plot_df = pd.pivot_table(
    df[(df["AGEGRP"] != "Not available") & (df["HHSIZE"] != "Not available")],
    values="WEIGHT",
    index="AGEGRP",
    columns="HHSIZE",
    aggfunc="sum",
)
sns.heatmap(plot_df, annot=False, cmap="Blues")  # cmap="flare")
# - male
plt.figure(figsize=fs)
plot_df = pd.pivot_table(
    df[
        (df["Gender"] == "Man+")
        & (df["AGEGRP"] != "Not available")
        & (df["HHSIZE"] != "Not available")
    ],
    values="WEIGHT",
    index="AGEGRP",
    columns="HHSIZE",
    aggfunc="sum",
)
sns.heatmap(plot_df, annot=False, cmap="Greens")  # cmap="flare")
# - female
plt.figure(figsize=fs)
plot_df = pd.pivot_table(
    df[
        (df["Gender"] == "Woman+")
        & (df["AGEGRP"] != "Not available")
        & (df["HHSIZE"] != "Not available")
    ],
    values="WEIGHT",
    index="AGEGRP",
    columns="HHSIZE",
    aggfunc="sum",
)
sns.heatmap(plot_df, annot=False, cmap="Purples")  # cmap="flare")


# %%

# employment income by hours of work, across sex

plt.figure(figsize=(20, 40))

df_processed = df[(df["EmpIn"] < 88888888)]
bins = list(range(0, 175000, 25000))
df_processed["EmpIn_cat"] = pd.cut(df_processed["EmpIn"], bins)
df_processed2 = df_processed[["Gender", "WKSWRK", "EmpIn_cat", "WEIGHT"]].reset_index()
plot_df = pd.pivot_table(
    df_processed,
    values="WEIGHT",
    index=["Gender", "WKSWRK", "EmpIn_cat"],
    columns=None,
    aggfunc="sum",
    observed=True,
).reset_index()
# sns.heatmap(plot_df, annot=False, cmap="Blues")  # cmap="flare")

fg = sns.FacetGrid(plot_df, col="Gender", col_wrap=3, height=6, aspect=0.75)


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop("data")
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs)


fg.map_dataframe(
    draw_heatmap,
    "EmpIn_cat",
    "WKSWRK",
    "WEIGHT",
    cmap="Blues",
    cbar=False,
    square=False,
)
# fg.set(yticks=[])
# fg.set(xticks=[])
plt.show()

# %%

# employment income, by work activity, across sex

plt.figure(figsize=(20, 40))

df_processed = df[(df["EmpIn"] < 88888888)]
bins = list(range(0, 175000, 25000))
df_processed["EmpIn_cat"] = pd.cut(df_processed["EmpIn"], bins)
df_processed2 = df_processed[["Gender", "WRKACT", "EmpIn_cat", "WEIGHT"]].reset_index()
plot_df = pd.pivot_table(
    df_processed,
    values="WEIGHT",
    index=["Gender", "WRKACT", "EmpIn_cat"],
    columns=None,
    aggfunc="sum",
    observed=True,
).reset_index()
# sns.heatmap(plot_df, annot=False, cmap="Blues")  # cmap="flare")

fg = sns.FacetGrid(plot_df, col="Gender", col_wrap=3, height=6, aspect=0.6)


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop("data")
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs)


fg.map_dataframe(
    draw_heatmap,
    "EmpIn_cat",
    "WRKACT",
    "WEIGHT",
    cmap="Blues",
    cbar=False,
    square=False,
)
# fg.set(yticks=[])
# fg.set(xticks=[])
plt.show()


# %%

# employment income, by employment occupations, across sex

plt.figure(figsize=(20, 40))

df_processed = df[(df["EmpIn"] < 88888888)]
df_processed = df_processed[
    ~(df_processed["NOCS_cat"].isin(["Not available", "Not applicable"]))
]
df_processed = df_processed.sort_values(["NOCS_cat", "AGEGRP"], ascending=True)

bins = list(range(0, 175000, 25000))
df_processed["EmpIn_cat"] = pd.cut(df_processed["EmpIn"], bins)
df_processed2 = df_processed[
    ["Gender", "NOCS_cat", "EmpIn_cat", "WEIGHT"]
].reset_index()
plot_df = pd.pivot_table(
    df_processed,
    values="WEIGHT",
    index=["Gender", "NOCS_cat", "EmpIn_cat"],
    columns=None,
    aggfunc="sum",
    observed=True,
).reset_index()
# sns.heatmap(plot_df, annot=False, cmap="Blues")  # cmap="flare")

fg = sns.FacetGrid(plot_df, col="Gender", col_wrap=3, height=6, aspect=0.8)


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop("data")
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs)


fg.map_dataframe(
    draw_heatmap,
    "EmpIn_cat",
    "NOCS_cat",
    "WEIGHT",
    cmap="Blues",
    cbar=False,
    square=False,
)
# fg.set(yticks=[])
# fg.set(xticks=[])
plt.show()


# %%

# income across gender (comparing against 2016)
df_processed_2016 = df_2016[["Sex", "EmpIn", "NOCS_cat", "WEIGHT"]].copy()
df_processed_2016["Year"] = "2016"
df_processed_2016 = df_processed_2016.rename(columns={"Sex": "Gender"})
df_processed_2016["Gender"] = df_processed_2016["Gender"].replace("Male", "Man+")
df_processed_2016["Gender"] = df_processed_2016["Gender"].replace("Female", "Woman+")
df_processed_2016 = df_processed_2016[~(df_processed_2016["NOCS_cat"].isin(["Not"]))]

df_processed = df[["Gender", "EmpIn", "NOCS_cat", "WEIGHT"]].copy()
df_processed["Year"] = "2021"
df_processed = df_processed[
    ~(df_processed["NOCS_cat"].isin(["Not available", "Not applicable"]))
]
df_processed = pd.concat([df_processed_2016, df_processed], axis=0)
df_processed = df_processed[(df_processed["EmpIn"] < 88888888)]
df_processed = df_processed[(df_processed["EmpIn"] != 1)]
df_processed = df_processed[(df_processed["EmpIn"] != -1)]
df_processed = df_processed[(df_processed["EmpIn"] != 0)]
df_processed = df_processed[(df_processed["EmpIn"] > 0)]
df_processed = df_processed[(df_processed["EmpIn"] < 250000)]
# df_processed = df_processed.sort_values(["NOCS_cat", "AGEGRP"], ascending=True)
# combine variable values
df_processed["Gender by Year"] = df_processed["Year"] + " " + df_processed["Gender"]


# plt.figure(figsize=(10, 5))
chart = sns.boxplot(
    data=df_processed,
    x="Gender by Year",
    y="EmpIn",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    fliersize=0.5,
    palette=sns.color_palette(),
    showmeans=True,
)
# chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")

# label medians
label_vals = df_processed.groupby(["Gender by Year"])["EmpIn"].median().astype(int)
label_txts = "s=" + label_vals.astype(str)
vertical_offset = (
    df_processed["EmpIn"].median().astype(int) * -0.05 * 3.5
)  # offset from median for display
for xtick in chart.get_xticks():
    chart.text(
        xtick,
        label_vals[xtick] + vertical_offset,
        label_txts[xtick],
        horizontalalignment="center",
        size="x-small",
        color="w",
        weight="semibold",
    )

# label means
label_vals = (
    df_processed.groupby(["Gender by Year"])["EmpIn"].mean().round(-2).astype(int)
)
label_txts = "M=" + label_vals.astype(str)
vertical_offset = (
    df_processed["EmpIn"].mean().round(-2).astype(int) * 0.05 * 1.5
)  # offset from median for display
for xtick in chart.get_xticks():
    chart.text(
        xtick,
        label_vals[xtick] + vertical_offset,
        label_txts[xtick],
        horizontalalignment="center",
        size="x-small",
        color="w",
        weight="semibold",
    )

# fix legend
sns.move_legend(chart, "upper right", title="Gender")

plt.show()


# %%

# employment status across gender
df_processed = df.copy()
df_processed = df_processed[
    ~(df_processed["NOCS_cat"].isin(["Not available", "Not applicable"]))
]
df_processed = df_processed[(df_processed["EmpIn"] < 88888888)]
df_processed = df_processed[(df_processed["EmpIn"] != 1)]
df_processed = df_processed[(df_processed["EmpIn"] != -1)]
df_processed = df_processed[(df_processed["EmpIn"] != 0)]
df_processed = df_processed[(df_processed["EmpIn"] > 0)]
df_processed = df_processed[(df_processed["EmpIn"] < 250000)]
df_processed = df_processed.sort_values(["NOCS_cat", "AGEGRP"], ascending=True)

# plt.figure(figsize=(10, 5))
chart = sns.boxplot(
    data=df_processed,
    x="NOCS_cat",
    y="EmpIn",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    fliersize=0.5,
    palette=sns.color_palette(),
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%

plt.figure(figsize=(10, 5))
chart = sns.lineplot(
    x="NOCS_cat", y="EmpIn", hue="Gender", data=df_processed, errorbar=("ci", 95)
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%

df_processed["WKSWRK"] = df_processed["WKSWRK"].astype(str)
plt.figure(figsize=(10, 5))
chart = sns.boxplot(
    data=df_processed[~(df_processed["WKSWRK"] == "Not available")].sort_values(
        ["WKSWRK"], ascending=True
    ),
    x="WKSWRK",
    y="EmpIn",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    fliersize=0.5,
    palette=sns.color_palette(),
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%

df_processed["WRKACT"] = df_processed["WRKACT"].astype(str)
plt.figure(figsize=(10, 5))
chart = sns.boxplot(
    data=df_processed[~(df_processed["WRKACT"] == "Not available")].sort_values(
        ["WRKACT"], ascending=True
    ),
    x="WRKACT",
    y="EmpIn",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    fliersize=0.5,
    palette=sns.color_palette(),
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%

plt.figure(figsize=(10, 5))
chart = sns.boxplot(
    data=df_processed[df_processed["VISMIN"] != "Not available"],
    x="VISMIN",
    y="EmpIn",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    fliersize=0.5,
    palette=sns.color_palette(),
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%

plt.figure(figsize=(10, 5))
chart = sns.boxplot(
    data=df_processed[
        ~df_processed["Marital"].isin(["Not available", "Not applicable"])
    ],
    x="Marital",
    y="EmpIn",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    fliersize=0.5,
    palette=sns.color_palette(),
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%

plt.figure(figsize=(10, 5))
chart = sns.boxplot(
    data=df_processed[~df_processed["PKIDS"].isin(["Not available", "Not applicable"])],
    x="PKIDS",
    y="EmpIn",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    fliersize=0.5,
    palette=sns.color_palette(),
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
# fix legend
sns.move_legend(chart, "upper right", title="Gender")
plt.show()

# %%

plt.figure(figsize=(10, 5))
chart = sns.boxplot(
    data=df_processed[df_processed["HHSIZE"] != "Not available"].sort_values(
        ["HHSIZE"], ascending=True
    ),
    x="HHSIZE",
    y="EmpIn",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    fliersize=0.5,
    palette=sns.color_palette(),
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
# fix legend
sns.move_legend(chart, "upper right", title="Gender")
plt.show()

# %%

plt.figure(figsize=(10, 5))

chart = sns.lineplot(
    x="HHSIZE",
    y="EmpIn",
    hue="Gender",
    data=df_processed[df_processed["HHSIZE"] != "Not available"],  # style="event",
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%

df_processed["AGEGRP"] = df_processed["AGEGRP"].astype(str)
plt.figure(figsize=(10, 5))
chart = sns.boxplot(
    data=df_processed[df_processed["AGEGRP"] != "Not available"].sort_values(
        ["AGEGRP"], ascending=True
    ),
    x="AGEGRP",
    y="EmpIn",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    fliersize=0.5,
    palette=sns.color_palette(),
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()


# %%

plt.figure(figsize=(10, 5))
chart = sns.lineplot(
    x="AGEGRP",
    y="EmpIn",
    hue="Gender",
    data=df_processed[df_processed["AGEGRP"] != "Not available"].sort_values(
        ["AGEGRP"], ascending=True
    ),
    errorbar=("ci", 95),
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%
# Household size and age

g = sns.relplot(
    data=df_processed[
        (df_processed["AGEGRP"] != "Not available")
        & (df_processed["HHSIZE4"] != "Not available")
    ].sort_values(["HHSIZE4", "AGEGRP"], ascending=True),
    x="AGEGRP",
    y="EmpIn",
    col="HHSIZE4",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    # style="event",
    kind="line",
    err_style="band",
    col_wrap=4,
    height=6,
    aspect=0.8,
    # palette="Set2",
    errorbar=("ci", 95),
)
g.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
g.add_legend()


# %%
# full time worker income by gender, across different number of weeks
df_processed2 = df_processed[
    (
        ~df_processed["WRKACT"].isin(
            [
                "Not available",
                "Not applicable",
                "Didn't work in 2015, worked in 2016",
                "Woked before 2015 or never worked",
            ]
        )
    )
    & (df_processed["WRKACT"].str.contains("full time"))
]
df_processed2["WRKACT"] = (
    df_processed2["WRKACT"]
    .astype(str)
    .replace({"Worked 1 to 13 weeks full time": "Worked 01 to 13 weeks full time"})
)
df_processed2 = df_processed2[df_processed2["AGEGRP"] != "Not available"]
df_processed2 = df_processed2.sort_values(["WRKACT", "AGEGRP"], ascending=True)
g = sns.relplot(
    data=df_processed2,
    x="AGEGRP",
    y="EmpIn",
    col="WRKACT",
    # col="HHSIZE",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    # style="event",
    kind="line",
    err_style="band",
    col_wrap=5,
    height=6,
    aspect=0.8,
    # palette="Set2",
    errorbar=("ci", 95),
)
for axes in g.axes.flat:
    _ = axes.set_xticklabels(
        axes.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
g.add_legend()


# %%
# part time worker income by gender, across different number of weeks
df_processed2 = df_processed[
    (
        ~df_processed["WRKACT"].isin(
            [
                "Not available",
                "Not applicable",
                "Didn't work in 2015, worked in 2016",
                "Woked before 2015 or never worked",
            ]
        )
    )
    & (df_processed["WRKACT"].str.contains("part time"))
]
df_processed2["WRKACT"] = (
    df_processed2["WRKACT"]
    .astype(str)
    .replace({"Worked 1 to 13 weeks full time": "Worked 01 to 13 weeks full time"})
)
df_processed2 = df_processed2[df_processed2["AGEGRP"] != "Not available"]
df_processed2 = df_processed2.sort_values(["WRKACT", "AGEGRP"], ascending=True)
g = sns.relplot(
    data=df_processed2,
    x="AGEGRP",
    y="EmpIn",
    col="WRKACT",
    # col="HHSIZE",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    # style="event",
    kind="line",
    err_style="band",
    col_wrap=5,
    height=6,
    aspect=0.8,
    # palette="Set2",
    errorbar=("ci", 95),
)
for axes in g.axes.flat:
    _ = axes.set_xticklabels(
        axes.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
g.add_legend()

# %%
# Weeks worked across occupation – All workers
df_processed2 = df_processed[
    (
        ~df_processed["WRKACT"].isin(
            [
                "Not available",
                "Not applicable",
                "Didn't work in 2015, worked in 2016",
                "Woked before 2015 or never worked",
            ]
        )
    )
]
df_processed2 = df_processed2[df_processed2["AGEGRP"] != "Not available"]
df_processed2 = df_processed2.sort_values(["WRKACT", "AGEGRP"], ascending=True)

g = sns.relplot(
    data=df_processed2,
    x="AGEGRP",
    y="EmpIn",
    col="NOCS_cat",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    # style="event",
    kind="line",
    col_wrap=5,
    height=6,
    aspect=0.8,
    # palette="Set2",
    errorbar=("ci", 95),
)
for axes in g.axes.flat:
    _ = axes.set_xticklabels(
        axes.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
g.add_legend()

# %%
# Weeks worked across occupation – Full time Full year
df_processed2 = df_processed[
    (
        ~df_processed["WRKACT"].isin(
            [
                "Not available",
                "Not applicable",
                "Didn't work in 2015, worked in 2016",
                "Woked before 2015 or never worked",
            ]
        )
    )
]
df_processed2 = df_processed2[
    (
        df_processed2["WRKACT"].isin(
            ["full time - 40 to 48 weeks", "full time - 49 to 52 weeks"]
        )
    )
]

g = sns.relplot(
    data=df_processed2,
    x="AGEGRP",
    y="EmpIn",
    col="NOCS_cat",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    # style="event",
    kind="line",
    col_wrap=5,
    height=6,
    aspect=0.8,
    # palette="Set2",
    errorbar=("ci", 95),
)
for axes in g.axes.flat:
    _ = axes.set_xticklabels(
        axes.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
g.add_legend()

# %%
# Weeks worked across occupation – Part time or Part year
# df_processed3 = df_processed2[df_processed2["WRKACT"] != "Worked 49 to 52 weeks full time"]
df_processed2 = df_processed.copy()
df_processed2 = df_processed2[
    ~(
        df_processed2["WRKACT"].isin(
            ["40 to 48 weeks full time", "49 to 52 weeks full time"]
        )
    )
]

g = sns.relplot(
    data=df_processed2,
    x="AGEGRP",
    y="EmpIn",
    col="NOCS_cat",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    # style="event",
    kind="line",
    col_wrap=5,
    height=6,
    aspect=0.8,
    # palette="Set2",
    errorbar=("ci", 95),
)
for axes in g.axes.flat:
    _ = axes.set_xticklabels(
        axes.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
g.add_legend()


# %%

# preprocess data
df_processed["AGEGRP"] = df_processed["AGEGRP"].astype(str)
df_processed["WRKACT"] = df_processed["WRKACT"].astype(str)
df_processed["WKSWRK"] = df_processed["WKSWRK"].astype(str)
df_processed2 = df_processed[df_processed["AGEGRP"] != -1]
df_processed2 = df_processed2[
    ~(df_processed2["NOCS_cat"].isin(["Not available", "Not applicable"]))
]
df_processed2 = df_processed2[
    ~(df_processed2["NOC21_cat"].isin(["Not available", "Not applicable"]))
]
df_processed2 = df_processed2[
    ~df_processed2["PKIDS"].isin(["Not available", "Not applicable"])
]
df_processed2 = df_processed2[(df_processed2["EmpIn"] < 88888888)]
df_processed2 = df_processed2[(df_processed2["EmpIn"] != 1)]
df_processed2 = df_processed2[(df_processed2["EmpIn"] != -1)]
df_processed2 = df_processed2[(df_processed2["EmpIn"] != 0)]
df_processed2 = df_processed2[(df_processed2["EmpIn"] > 0)]
df_processed2 = df_processed2[(df_processed2["EmpIn"] < 250000)]
df_processed2 = df_processed2[~(df_processed2["AGEGRP"] == "Not available")]
df_processed2 = df_processed2[~(df_processed2["HHSIZE"] == "Not available")]
df_processed2 = df_processed2[~(df_processed2["HHSIZE4"] == "Not available")]
df_processed2 = df_processed2[~(df_processed2["WKSWRK"] == "Not available")]
df_processed2 = df_processed2[~(df_processed2["Marital"] == "Not available")]
df_processed2 = df_processed2[~(df_processed2["VISMIN"] == "Not available")]
df_processed2 = df_processed2[~(df_processed2["HighestDegrees"] == "Not available")]
df_processed2 = df_processed2[
    (
        ~df_processed2["WRKACT"].isin(
            [
                "Not available",
                "Not applicable",
                "Didn't work in 2015, worked in 2016",
                "Woked before 2015 or never worked",
                "Did not work in 2020",
            ]
        )
    )
    # & (df_processed2["WRKACT"].str.contains("full time"))
]
df_processed2["WRKACT"] = (
    df_processed2["WRKACT"]
    .astype(str)
    .replace({"Worked 1 to 13 weeks full time": "Worked 01 to 13 weeks full time"})
)
df_processed3 = df_processed2.sort_values(["NOCS_cat", "AGEGRP"], ascending=True)

# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf

df_processed3["EmpIncomeK"] = df_processed3["EmpIn"] / 1000
df_processed3["AGEGRP"] = df_processed3["AGEGRP"].replace("10 to 17", "")
formula = "EmpIncomeK ~ VISMIN + HighestDegrees + C(AGEGRP)  + NOCS_cat + WRKACT + Gender*PKIDS + Gender*Marital + Gender"
results = smf.ols(
    formula,
    data=df_processed3,
).fit()

# results.summary()

# # %%
# # df_pgrid = df_processed3[["Gender", "AGEGRP", "EmpIncomeK", "VISMIN", "HighestDegrees"]]
# df_pgrid = df_processed3[["Gender", "EmpIncomeK", "AGEGRP"]]#, "EmpIncomeK"]]
# g = sns.PairGrid(df_pgrid, hue="Gender", vars=["EmpIncomeK", "AGEGRP"])
# g.map(sns.swarmplot)

# %%
xname = [
    "intercept",
    "not_visible_minority",
    "degree_diplomas",
    "degree_bachelor",
    "degree_masters",
    "degree_doctorates",
    "age_18to24",
    "age_25to34",
    "age_35to44",
    "age_45to54",
    "age_55to64",
    "age_65to74",
    "age_75plus",
    "noc_1_business",
    "noc_2_sciences",
    "noc_3_health",
    "noc_4_edu&law&govt",
    "noc_5_art&culture&rec",
    "noc_6_sales&services",
    "noc_7_trades&transport",
    "noc_8_resources&agriculture",
    "noc_9_manufacturing&utilities",
    "fulltime_14to26weeks",
    "fulltime_27to39weeks",
    "fulltime_40to48weeks",
    "fulltime_49to52weeks",
    "parttime_01to13weeks",
    "parttime_14to26weeks",
    "parttime_27to39weeks",
    "parttime_40to48weeks",
    "parttime_49to52weeks",
    "woman",
    "hh_has_kids",
    "single_or_separated",
    "woman * hh_has_kids",
    "woman * single_or_separated",
]
results.summary(xname=xname)

# %%
# results.cov_params()

# %%
df_processed3 = df_processed3.sort_values(
    ["Gender", "Marital_Status", "PKIDS"], ascending=True
)
# df_processed3.groupby(["Gender", "Marital_Status", "PKIDS"]).count()["EmpIncomeK"]
g = sns.FacetGrid(df_processed3, col="Marital_Status", row="PKIDS", margin_titles=True)
g.map_dataframe(
    sns.histplot,
    x="EmpIncomeK",
    hue="Gender",
    common_norm=False,
    fill=True,
    alpha=0.5,
    linewidth=0.7,
)
g.add_legend()

# %%
df_processed3 = df_processed3.sort_values(
    ["Gender", "HHSIZE4", "Marital"], ascending=True
)
g = sns.FacetGrid(df_processed3, col="HHSIZE4", row="Marital", margin_titles=True)
g.map_dataframe(
    sns.kdeplot,
    x="EmpIncomeK",
    hue="Gender",
    common_norm=False,
    fill=True,
    alpha=0.5,
    linewidth=0.7,
)
g.add_legend()

# %%
df_processed3 = df_processed3.sort_values(
    ["Gender", "VISMIN", "Marital"], ascending=True
)
g = sns.FacetGrid(df_processed3, col="Marital", row="VISMIN", margin_titles=True)
g.map_dataframe(
    sns.kdeplot,
    x="EmpIncomeK",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    common_norm=False,
    fill=True,
    alpha=0.5,
    linewidth=0.7,
)
g.add_legend()

# %%
df_processed3 = df_processed3.sort_values(["Gender", "VISMIN", "PKIDS"], ascending=True)
g = sns.FacetGrid(df_processed3, col="PKIDS", row="VISMIN", margin_titles=True)
g.map_dataframe(
    sns.kdeplot,
    x="EmpIncomeK",
    hue="Gender",
    hue_order=["Man+", "Woman+"],
    common_norm=False,
    fill=True,
    alpha=0.5,
    linewidth=0.7,
)
g.add_legend()

# %%
