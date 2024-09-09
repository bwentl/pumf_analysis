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

# StandardScaler preferred over Normalizer, see note 1
# note 1: https://datascience.stackexchange.com/questions/45900/when-to-use-standard-scaler-and-when-normalizer


# %%

df = pumf_2016_util.read_indiv_table()
pumf_2016_util.recode_indiv_table(df)

# list columns
print(list(df.columns))

df_2016 = df.copy()
del df



# %%
# # %%

# preloaded_fea_file = "data/census_pumf_vancouver_2021.fea"
# data_path = "../data_packages/PUMF_Census_2021_StatCan/ind/data_donnees_2021_ind_v2.csv"


# if not os.path.exists(preloaded_fea_file):
#     df = pd.read_csv(data_path)
#     # import pyreadstat
#     # df_2021, meta = pyreadstat.read_dta(data_path)
#     # filter for CMA equals to Vancouver
#     # source: https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=317043&CVD=317046&CPV=59A&CST=01012016&CLV=3&MLV=5
#     # df = df[df["CMA"] == "Vancouver"].reset_index(
#     #     drop=True
#     # )  # filter by Vancouver CMA only
#     # save to feather to allow faster loading later
#     df.to_feather(preloaded_fea_file)
# else:
#     df = pd.read_feather(preloaded_fea_file)

# # list columns
# print(list(df.columns))

# df_2021 = df.copy()
# del df

# %%

# population by age

plot_df = pd.pivot_table(
    df, values="WEIGHT", index="AGEBIN", columns=None, aggfunc="sum"
).reset_index()

plt.figure(figsize=(12, 6))
chart = sns.barplot(data=plot_df, x="AGEBIN", y="WEIGHT", palette="Paired")
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%

# population of age by household size

plt.figure(figsize=(8, 7))
plot_df = pd.pivot_table(
    df, values="WEIGHT", index="AGEBIN", columns="HHSIZE", aggfunc="sum"
)
sns.heatmap(plot_df, annot=False, cmap="Blues")  # cmap="flare")

# %%

# population of age by household size, Male

plt.figure(figsize=(8, 7))
plot_df = pd.pivot_table(
    df[df["Sex"] == "Male"],
    values="WEIGHT",
    index="AGEBIN",
    columns="HHSIZE",
    aggfunc="sum",
)
sns.heatmap(plot_df, annot=False, cmap="Greens")  # cmap="flare")


# %%

# population of age by household size, Female

plt.figure(figsize=(8, 7))
plot_df = pd.pivot_table(
    df[df["Sex"] == "Female"],
    values="WEIGHT",
    index="AGEBIN",
    columns="HHSIZE",
    aggfunc="sum",
)
sns.heatmap(plot_df, annot=False, cmap="Purples")  # cmap="flare")


# %%

# mobility status (moving) 5 year by household size

plot_df = pd.pivot_table(
    df, values="WEIGHT", index="Mob5", columns="HHSIZE", aggfunc="sum"
)
sns.heatmap(plot_df, annot=False, cmap="Blues")  # cmap="flare")


# %%

# mobility status (moving) 5 year by household size, exclude non movers

plot_df = pd.pivot_table(
    df[~df["Mob5"].isin(["Non-migrants", "Non-movers", "Not applicable"])],
    values="WEIGHT",
    index="Mob5",
    columns="HHSIZE",
    aggfunc="sum",
    observed=True,
)
sns.heatmap(plot_df, annot=False, cmap="Blues")  # cmap="flare")


# %%

# total income by household size

df_processed = df[(df["EmpIn"] < 88888888)]
bins = list(range(0, 175000, 25000))
df_processed["EmpIn_cat"] = pd.cut(df_processed["EmpIn"], bins)
plot_df = pd.pivot_table(
    df_processed,
    values="WEIGHT",
    index="HHSIZE",
    columns="EmpIn_cat",
    aggfunc="sum",
    observed=True,
)
sns.heatmap(plot_df, annot=False, cmap="Blues")  # cmap="flare")

# %%

# total income by household size, with children only

df_processed = df[(df["EmpIn"] < 88888888)]
df_processed = df_processed[(df_processed["PKIDS"] == "One or more")]
bins = list(range(0, 175000, 25000))
df_processed["EmpIn_cat"] = pd.cut(df_processed["EmpIn"], bins)
plot_df = pd.pivot_table(
    df_processed,
    values="WEIGHT",
    index="HHSIZE",
    columns="EmpIn_cat",
    aggfunc="sum",
    observed=True,
)
sns.heatmap(plot_df, annot=False, cmap="Blues")  # cmap="flare")


# %%

# employment income by hours of work, across sex

plt.figure(figsize=(20, 40))

df_processed = df[(df["EmpIn"] < 88888888)]
bins = list(range(0, 175000, 25000))
df_processed["EmpIn_cat"] = pd.cut(df_processed["EmpIn"], bins)
df_processed2 = df_processed[["Sex", "WKSWRK", "EmpIn_cat", "WEIGHT"]].reset_index()
plot_df = pd.pivot_table(
    df_processed,
    values="WEIGHT",
    index=["Sex", "WKSWRK", "EmpIn_cat"],
    columns=None,
    aggfunc="sum",
    observed=True,
).reset_index()
# sns.heatmap(plot_df, annot=False, cmap="Blues")  # cmap="flare")

fg = sns.FacetGrid(plot_df, col="Sex", col_wrap=3, height=6, aspect=0.75)


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
df_processed2 = df_processed[["Sex", "WRKACT", "EmpIn_cat", "WEIGHT"]].reset_index()
plot_df = pd.pivot_table(
    df_processed,
    values="WEIGHT",
    index=["Sex", "WRKACT", "EmpIn_cat"],
    columns=None,
    aggfunc="sum",
    observed=True,
).reset_index()
# sns.heatmap(plot_df, annot=False, cmap="Blues")  # cmap="flare")

fg = sns.FacetGrid(plot_df, col="Sex", col_wrap=3, height=6, aspect=0.6)


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
bins = list(range(0, 175000, 25000))
df_processed["EmpIn_cat"] = pd.cut(df_processed["EmpIn"], bins)
df_processed2 = df_processed[["Sex", "NOCS", "EmpIn_cat", "WEIGHT"]].reset_index()
plot_df = pd.pivot_table(
    df_processed,
    values="WEIGHT",
    index=["Sex", "NOCS", "EmpIn_cat"],
    columns=None,
    aggfunc="sum",
    observed=True,
).reset_index()
# sns.heatmap(plot_df, annot=False, cmap="Blues")  # cmap="flare")

fg = sns.FacetGrid(plot_df, col="Sex", col_wrap=3, height=6, aspect=0.8)


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop("data")
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs)


fg.map_dataframe(
    draw_heatmap,
    "EmpIn_cat",
    "NOCS",
    "WEIGHT",
    cmap="Blues",
    cbar=False,
    square=False,
)
# fg.set(yticks=[])
# fg.set(xticks=[])
plt.show()


# %%

# employment status across gender
df_processed = df.copy()
df_processed = df_processed[(df_processed["NOCS_cat"] != "Not")]
df_processed = df_processed[(df_processed["EmpIn"] < 88888888)]
df_processed = df_processed[(df_processed["EmpIn"] != 1)]
df_processed = df_processed[(df_processed["EmpIn"] != -1)]
df_processed = df_processed[(df_processed["EmpIn"] != 0)]
df_processed = df_processed[(df_processed["EmpIn"] > 0)]
df_processed = df_processed[(df_processed["EmpIn"] < 250000)]
df_processed = df_processed.sort_values(["NOCS_cat", "AGEBIN"], ascending=True)

plt.figure(figsize=(10, 5))
chart = sns.violinplot(
    data=df_processed,
    x="NOCS",
    y="EmpIn",
    hue="Sex",
    hue_order=["Male", "Female"],
    split=True,
    inner="quart",
    cut=0,
    palette=sns.color_palette(),
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%

plt.figure(figsize=(10, 5))
chart = sns.lineplot(
    x="NOCS", y="EmpIn", hue="Sex", data=df_processed, errorbar=("ci", 95)
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%

plt.figure(figsize=(10, 5))
chart = sns.violinplot(
    data=df_processed,
    x="WKSWRK",
    y="EmpIn",
    hue="Sex",
    hue_order=["Male", "Female"],
    split=True,
    inner="quart",
    cut=0,
    palette=sns.color_palette(),
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%

plt.figure(figsize=(10, 5))
chart = sns.violinplot(
    data=df_processed,
    x="WRKACT",
    y="EmpIn",
    hue="Sex",
    hue_order=["Male", "Female"],
    split=True,
    inner="quart",
    cut=0,
    palette=sns.color_palette(),
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%

plt.figure(figsize=(10, 5))
chart = sns.violinplot(
    data=df_processed[df_processed["HHSIZE"] != "Not available"],
    x="HHSIZE",
    y="EmpIn",
    hue="Sex",
    hue_order=["Male", "Female"],
    split=True,
    inner="quart",
    cut=0,
    palette=sns.color_palette(),
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%

plt.figure(figsize=(10, 5))

chart = sns.lineplot(
    x="HHSIZE",
    y="EmpIn",
    hue="Sex",
    data=df_processed[df_processed["HHSIZE"] != "Not available"],  # style="event",
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%

df_processed2 = df_processed[df_processed["AGE"] != -1]
df_processed2["AGEBIN"] = df_processed2["AGEBIN"].astype(str)
df_processed2 = df_processed2.sort_values(["AGEBIN"], ascending=True)
plt.figure(figsize=(10, 5))
chart = sns.violinplot(
    data=df_processed2,
    x="AGEBIN",
    y="EmpIn",
    hue="Sex",
    hue_order=["Male", "Female"],
    split=True,
    inner="quart",
    cut=0,
    palette=sns.color_palette(),
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()


# %%

plt.figure(figsize=(10, 5))
chart = sns.lineplot(
    x="AGEBIN", y="EmpIn", hue="Sex", data=df_processed2, errorbar=("ci", 95)
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

# %%
# Household size and age

df_processed3 = df_processed2[~(df_processed2["HHSIZE"] == "Not available")]
df_processed3["HHSIZE4"] = df_processed3["HHSIZE"].astype(str)
df_processed3["HHSIZE4"] = df_processed3["HHSIZE4"].map(
    {
        "1 person": "1 person",
        "2 persons": "2 persons",
        "3 persons": "3 persons",
        "4 persons": "4+ persons",
        "5 persons": "4+ persons",
        "6 persons": "4+ persons",
        "7 persons": "4+ persons",
    }
)
df_processed3 = df_processed3.sort_values(["HHSIZE4", "AGEBIN"], ascending=True)

g = sns.relplot(
    data=df_processed3,
    x="AGEBIN",
    y="EmpIn",
    col="HHSIZE4",
    hue="Sex",
    hue_order=["Male", "Female"],
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
df_processed3 = df_processed2[
    (
        ~df_processed2["WRKACT"].isin(
            [
                "Not available",
                "Not applicable",
                "Didn't work in 2015, worked in 2016",
                "Woked before 2015 or never worked",
            ]
        )
    )
    & (df_processed2["WRKACT"].str.contains("full time"))
]
df_processed3["WRKACT"] = (
    df_processed3["WRKACT"]
    .astype(str)
    .replace({"Worked 1 to 13 weeks full time": "Worked 01 to 13 weeks full time"})
)
df_processed3 = df_processed3.sort_values(["WRKACT", "AGEBIN"], ascending=True)
g = sns.relplot(
    data=df_processed3,
    x="AGEBIN",
    y="EmpIn",
    col="WRKACT",
    # col="HHSIZE",
    hue="Sex",
    hue_order=["Male", "Female"],
    # style="event",
    kind="line",
    err_style="band",
    col_wrap=5,
    height=6,
    aspect=0.8,
    # palette="Set2",
    errorbar=("ci", 95),
)
g.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
g.add_legend()


# %%
# part time worker income by gender, across different number of weeks
df_processed3 = df_processed2[
    (
        ~df_processed2["WRKACT"].isin(
            [
                "Not available",
                "Not applicable",
                "Didn't work in 2015, worked in 2016",
                "Woked before 2015 or never worked",
            ]
        )
    )
    & (df_processed2["WRKACT"].str.contains("part time"))
]
df_processed3["WRKACT"] = (
    df_processed3["WRKACT"]
    .astype(str)
    .replace({"Worked 1 to 13 weeks full time": "Worked 01 to 13 weeks full time"})
)
df_processed3 = df_processed3.sort_values(["WRKACT", "AGEBIN"], ascending=True)
g = sns.relplot(
    data=df_processed3,
    x="AGEBIN",
    y="EmpIn",
    col="WRKACT",
    # col="HHSIZE",
    hue="Sex",
    hue_order=["Male", "Female"],
    # style="event",
    kind="line",
    err_style="band",
    col_wrap=5,
    height=6,
    aspect=0.8,
    # palette="Set2",
    errorbar=("ci", 95),
)
g.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
g.add_legend()

# %%
# Weeks worked across occupation – All workers
df_processed2 = df_processed2.sort_values(["NOCS_cat", "AGEBIN"], ascending=True)

g = sns.relplot(
    data=df_processed2,
    x="AGEBIN",
    y="EmpIn",
    col="NOCS_cat",
    hue="Sex",
    hue_order=["Male", "Female"],
    # style="event",
    kind="line",
    col_wrap=5,
    height=6,
    aspect=0.8,
    # palette="Set2",
    errorbar=("ci", 95),
)
g.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
g.add_legend()

# %%
# Weeks worked across occupation – Full time Full year
# df_processed3 = df_processed2[df_processed2["WRKACT"] != "Worked 49 to 52 weeks full time"]
df_processed3 = df_processed2.copy()
df_processed3 = df_processed3[
    (
        df_processed3["WRKACT"].isin(
            ["Worked 40 to 48 weeks full time", "Worked 49 to 52 weeks full time"]
        )
    )
]

g = sns.relplot(
    data=df_processed3,
    x="AGEBIN",
    y="EmpIn",
    col="NOCS_cat",
    hue="Sex",
    hue_order=["Male", "Female"],
    # style="event",
    kind="line",
    col_wrap=5,
    height=6,
    aspect=0.8,
    # palette="Set2",
    errorbar=("ci", 95),
)
g.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
g.add_legend()

# %%
# Weeks worked across occupation – Part time or Part year
# df_processed3 = df_processed2[df_processed2["WRKACT"] != "Worked 49 to 52 weeks full time"]
df_processed3 = df_processed2.copy()
df_processed3 = df_processed3[
    ~(
        df_processed3["WRKACT"].isin(
            ["Worked 40 to 48 weeks full time", "Worked 49 to 52 weeks full time"]
        )
    )
]

g = sns.relplot(
    data=df_processed3,
    x="AGEBIN",
    y="EmpIn",
    col="NOCS_cat",
    hue="Sex",
    hue_order=["Male", "Female"],
    # style="event",
    kind="line",
    col_wrap=5,
    height=6,
    aspect=0.8,
    # palette="Set2",
    errorbar=("ci", 95),
)
g.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
g.add_legend()


# %%
# Weeks worked across occupation – Part time or Part year - across weeks worked
# df_processed3 = df_processed2[df_processed2["WRKACT"] != "Worked 49 to 52 weeks full time"]
df_processed3 = df_processed2.copy()
df_processed3 = df_processed3[
    ~(
        df_processed3["WRKACT"].isin(
            ["Worked 40 to 48 weeks full time", "Worked 49 to 52 weeks full time"]
        )
    )
]
df_processed3["WRKACT"] = df_processed3["WRKACT"].astype(str)

g = sns.relplot(
    data=df_processed3,
    x="NOCS_cat",
    y="EmpIn",
    col="WRKACT",
    hue="Sex",
    hue_order=["Male", "Female"],
    # style="event",
    kind="line",
    col_wrap=5,
    height=6,
    aspect=0.8,
    # palette="Set2",
    errorbar=("ci", 95),
)
g.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")
g.add_legend()


# %%

# Full time part time may be a predictor for woman's income
#


# %%
# Analysis of wages
# LFACT == 1
df = df[(df.Wages < 88888888) & (df.LFACT == "Employed - Worked in reference week")]
df = df.reset_index(drop=True)

# %%
sns.boxplot(data=df, x="Sex", y="Wages")

# %%
sns.kdeplot(data=df, hue="Sex", x="Wages", bw_adjust=1.7)

# %%
sns.scatterplot(data=df, x="Wages", y="AGE")
sns.rugplot(data=df, x="Wages", y="AGE")

# %%
sns.kdeplot(data=df, x="Wages", hue="Sex")
sns.rugplot(data=df, x="Wages", hue="Sex")

# %%
df[df.Wages < 150000].plot.hexbin(x="Wages", y="AGE", gridsize=20)

# %%
wages_mobility = pd.pivot_table(
    df, index="Mob5", columns="MOB1", values="Wages", aggfunc="mean"
)
sns.heatmap(wages_mobility, annot=False, cmap="crest", vmin=30000, vmax=100000)

# %%
print(list(df.columns))

# %% [markdown]
# Regression Models

# %%
# variable recoding

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# https://stackoverflow.com/questions/58101126/using-scikit-learn-onehotencoder-with-a-pandas-dataframe
# # try OneHotEncoder
# oht_encode = OneHotEncoder()
# oht_encode.fit(df['MOB5_cat'].unique().reshape(-1, 1))
# transformed = oht_encode.transform(
#     df['MOB5_cat'].to_numpy().reshape(-1, 1)).toarray()
# ohe_df = pd.DataFrame(transformed, columns=oht_encode.get_feature_names_out())
# df = pd.concat([df, ohe_df], axis=1).drop(['MOB5_cat'], axis=1)
# do the same thing with get dummies
df = pd.get_dummies(df, prefix=["MOB5_cat"], columns=["MOB5_cat"], drop_first=False)
df = pd.get_dummies(df, prefix=["MOB1_cat"], columns=["MOB1_cat"], drop_first=False)
df = pd.get_dummies(df, prefix=["NOCS_cat"], columns=["NOCS_cat"], drop_first=False)

# %%
# variable selection
# y - Wages
# X
# - MOB1_cat: moved in the last year
# - MOB5_cat: moved in the last 5 years
# "HHSIZE_int", "Sex_cat", "BedRm_cat", "MOB5_cat_moved_to_canada_5", "MOB5_cat_moved_within_canada_5", "MOB5_cat_not_moved_5", "MOB1_cat_moved_to_canada_1", "MOB1_cat_moved_within_canada_1", "MOB1_cat_not_moved_1"

# A1. simple regression model
pred_var = "Wages"
variable_list = [
    "AGE",
    "Sex_cat",
    "BedRm_cat",
    "MOB5_cat_moved_to_canada_5",
    "MOB5_cat_moved_within_canada_5",
    # "MOB5_cat_not_moved_5",
    # "MOB1_cat_moved_to_canada_1",
    # "MOB1_cat_moved_within_canada_1",
    "MOB1_cat_not_moved_1",
    "NOCS_cat_Not",
    "NOCS_cat_art_trades_manuf",
    # "NOCS_cat_busi",
    "NOCS_cat_health_edu_law",
    "NOCS_cat_mgmt_sci",
    "NOCS_cat_sales_agri",
]

X, y = df[variable_list], df[pred_var]

X = sm.add_constant(X)
variable_list.append("const")
model = sm.OLS(y, X).fit()
print(model.summary())

# %%

# A2. Simple regression model with formula (simple OLS)
# using formula is very convenient for specifying different models
# https://www.statsmodels.org/stable/api.html#statsmodels-formula-api
# print(' + '.join(variable_list))
results = smf.ols(
    "Wages ~ np.log(AGE) + Sex_cat + BedRm_cat + MOB5_cat_moved_to_canada_5 + MOB5_cat_moved_within_canada_5 + MOB1_cat_not_moved_1 + NOCS_cat_Not + NOCS_cat_art_trades_manuf + NOCS_cat_health_edu_law + NOCS_cat_mgmt_sci + NOCS_cat_sales_agri",
    data=df,
).fit()
print(results.summary())

# %%

# read: https://machinelearningresearch.quora.com/What-is-difference-between-ordinary-linear-regression-and-generalized-linear-model-or-linear-regression

# A3. Generalize Linear
results = smf.gls(
    "Wages ~ np.log(AGE) + Sex_cat + BedRm_cat + MOB5_cat_moved_to_canada_5 + MOB5_cat_moved_within_canada_5 + MOB1_cat_not_moved_1 + NOCS_cat_Not + NOCS_cat_art_trades_manuf + NOCS_cat_health_edu_law + NOCS_cat_mgmt_sci + NOCS_cat_sales_agri",
    data=df,
).fit()
print(results.summary())

# %% [markdown]
# # Model options from StatsModel
# - OLS: Ordinary Least Square - https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
# - WLS: Weighted Least Square - https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.WLS.html#statsmodels.regression.linear_model.WLS
# - GLS: Generalized Least Square - https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLS.html#statsmodels.regression.linear_model.GLS
# - GLSAR: Generalized Least Squares with AR covariance structure - specify covariance structure
# - MixedLM: Linear Mixed Effects - https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html#statsmodels.regression.mixed_linear_model.MixedLM
# - GEE, Ordinal GEE, Nominal GEE: Estimating equations / structural equations
# - RLM: Robust linear model - https://www.statsmodels.org/stable/generated/statsmodels.robust.robust_linear_model.RLM.html#statsmodels.robust.robust_linear_model.RLM
# Discrete Choices - https://www.statsmodels.org/stable/examples/notebooks/generated/discrete_choice_overview.html
# - Logit: Logistic regression - https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html#statsmodels.discrete.discrete_model.Logit
# - Probit: Probit regression - https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Probit.html#statsmodels.discrete.discrete_model.Probit
# - MNLogit: Multinomial logit regression - https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.MNLogit.html#statsmodels.discrete.discrete_model.MNLogit
# - Poisson: Poisson discrete choice - https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Poisson.html#statsmodels.discrete.discrete_model.Poisson
# - Negative Binomial discrete choice - https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.NegativeBinomial.html#statsmodels.discrete.discrete_model.NegativeBinomial
# Misc:
# - Quantile Regression - https://www.statsmodels.org/stable/generated/statsmodels.regression.quantile_regression.QuantReg.html#statsmodels.regression.quantile_regression.QuantReg
# - Hazard Regression - https://www.statsmodels.org/stable/generated/statsmodels.duration.hazard_regression.PHReg.html#statsmodels.duration.hazard_regression.PHReg
# - GLM Additive Model - https://www.statsmodels.org/stable/generated/statsmodels.gam.generalized_additive_model.GLMGam.html#statsmodels.gam.generalized_additive_model.GLMGam

# feature selection
# - Variance Threshold
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
sel.fit(X)
print(sel.get_feature_names_out())


# %%
import math

# B1: ML Regression
# https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

# OLS Linear Regression using sklearn
reg = linear_model.LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)
print(reg.intercept_, reg.coef_, reg.score(X, y))
print(f"r2: {r2_score(y, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y, y_pred))}")

# %%
# Ridge Regression
reg = linear_model.Ridge(alpha=0.5)
reg.fit(X, y)
y_pred = reg.predict(X)
print(reg.intercept_, reg.coef_, reg.score(X, y))
print(f"r2: {r2_score(y, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y, y_pred))}")

# %%
# Lasso Regression
reg = linear_model.Lasso(alpha=0.1)
reg.fit(X, y)
y_pred = reg.predict(X)
print(reg.intercept_, reg.coef_, reg.score(X, y))
print(f"r2: {r2_score(y, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y, y_pred))}")

# %%
# Elastic Net with Cross Grid
reg = linear_model.ElasticNetCV(cv=5, random_state=0)
reg.fit(X, y)
y_pred = reg.predict(X)
print(reg.alpha_)
print(reg.intercept_, reg.coef_, reg.score(X, y))
print(f"r2: {r2_score(y, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y, y_pred))}")

# # %%
# # Support Vector Machine for Regression
# from sklearn import svm

# reg = svm.SVR()
# reg.fit(X, y)
# y_pred = reg.predict(X)
# print(reg.alpha_)
# print(reg.intercept_, reg.coef_, reg.score(X, y))
# print(f"r2: {r2_score(y, y_pred)}")
# print(f"rmse: {math.sqrt(mean_squared_error(y, y_pred))}")

# %%
# Regression Tree
# https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html

# normalize data
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
transformer = Normalizer().fit(X)
Xt = transformer.transform(X)

reg = tree.DecisionTreeRegressor(max_depth=6)
reg.fit(Xt, y)
y_pred = reg.predict(Xt)
print(f"r2: {r2_score(y, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y, y_pred))}")
# tree.plot_tree(reg)
# https://stackoverflow.com/questions/68352933/name-of-variables-in-sklearn-pipeline
r = tree.export_text(reg, feature_names=variable_list)
print(r)
pd.DataFrame({"name": list(X.columns), "value": reg.feature_importances_}).head(20)

# %%
# Random Forest
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = RandomForestClassifier(n_estimators=5)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"r2: {r2_score(y_test, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y_test, y_pred))}")
pd.DataFrame({"name": list(X.columns), "value": reg.feature_importances_}).head(20)


# %%
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = AdaBoostClassifier(n_estimators=5)
scores = cross_val_score(reg, X_train, y_train, cv=5)
print(f"scores: {scores}")
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"r2: {r2_score(y_test, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y_test, y_pred))}")
pd.DataFrame({"name": list(X.columns), "value": reg.feature_importances_}).head(20)


# %%
# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = GradientBoostingRegressor(random_state=0)
scores = cross_val_score(reg, X_train, y_train, cv=5)
print(f"scores: {scores}")
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"r2: {r2_score(y_test, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y_test, y_pred))}")
pd.DataFrame({"name": list(X.columns), "value": reg.feature_importances_}).head(20)

# %%
# Histogram-based Gradient Boosting
# https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
from sklearn.ensemble import HistGradientBoostingRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = HistGradientBoostingRegressor(random_state=0)
scores = cross_val_score(reg, X_train, y_train, cv=5)
print(f"scores: {scores}")
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"r2: {r2_score(y_test, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y_test, y_pred))}")
# pd.DataFrame({"name": list(X.columns), "value": reg.feature_importances_}).head(20)

# %%
# # Artificial Neural Network
# from sklearn.neural_network import MLPClassifier

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# reg = MLPClassifier(
#     solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
# )
# scores = cross_val_score(reg, X_train, y_train, cv=5)
# print(f"scores: {scores}")
# reg.fit(X_train, y_train)
# y_pred = reg.predict(X_test)
# print(f"r2: {r2_score(y_test, y_pred)}")
# print(f"rmse: {math.sqrt(mean_squared_error(y_test, y_pred))}")

# # ~25 mins

# %% [markdown]
# Classification Models

# %%
# C1: ML Classification

pred_var = "BedRm"
variable_list = [
    "AGE",
    "Wages",
    "Sex_cat",
    "MOB5_cat_moved_to_canada_5",
    "MOB5_cat_moved_within_canada_5",
    # "MOB5_cat_not_moved_5",
    # "MOB1_cat_moved_to_canada_1",
    # "MOB1_cat_moved_within_canada_1",
    "MOB1_cat_not_moved_1",
    "NOCS_cat_Not",
    "NOCS_cat_art_trades_manuf",
    # "NOCS_cat_busi",
    "NOCS_cat_health_edu_law",
    "NOCS_cat_mgmt_sci",
    "NOCS_cat_sales_agri",
]

X, y = df[variable_list], df[pred_var]
y_classes = list(df[pred_var].unique())

# %%
# Classification Tree
# https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# normalize data
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
transformer = Normalizer().fit(X)
Xt = transformer.transform(X)

reg = tree.DecisionTreeClassifier(max_depth=6)
reg.fit(Xt, y)
y_pred = reg.predict(Xt)
print(f"score: {accuracy_score(y, y_pred)}")
print(classification_report(y, y_pred, target_names=y_classes))
# tree.plot_tree(reg)
# https://stackoverflow.com/questions/68352933/name-of-variables-in-sklearn-pipeline
r = tree.export_text(reg, feature_names=variable_list)
print(r)
pd.DataFrame({"name": list(X.columns), "value": reg.feature_importances_}).head(20)

# notes:
# Precision - TP / (TP + FP): fraction of true that is actually true
# Recall - TP / (TP + FN): fraction of true that was originally true
# f score - composite (avg) of precision and recall, 1 is perfect


# %% [markdown]
# Clustering Models

variable_list = [
    "AGE",
    "Wages",
    "Sex_cat",
    "MOB5_cat_moved_to_canada_5",
    "MOB5_cat_moved_within_canada_5",
    # "MOB5_cat_not_moved_5",
    # "MOB1_cat_moved_to_canada_1",
    # "MOB1_cat_moved_within_canada_1",
    "MOB1_cat_not_moved_1",
    "NOCS_cat_Not",
    "NOCS_cat_art_trades_manuf",
    # "NOCS_cat_busi",
    "NOCS_cat_health_edu_law",
    "NOCS_cat_mgmt_sci",
    "NOCS_cat_sales_agri",
]

X = df[variable_list]

# %%
# D1: ML Clustering

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
kmeans.labels_
y = kmeans.predict(X)
cc = kmeans.cluster_centers_
# dict(zip(list(range(0,len(cc))), cc))
cc_dict = {("cluster" + str(i)): cc[i] for i in range(0, len(cc))}
view_dict = {"name": variable_list}
view_dict.update(cc_dict)
pd.DataFrame(view_dict)

# %%
df["cluster"] = y
df.groupby("cluster").sum()["WEIGHT"].round(0) / df.sum()["WEIGHT"] * 100

# %%

# sns.scatterplot(data=df.sample(100), x="Wages", y="AGE", hue="cluster", size=10)

sns.displot(
    data=df.sample(100), x="Wages", y="AGE", hue="cluster", kind="kde", rug=True
)

# sns.displot(
#     data=df.sample(100), x="Wages", y="AGE", hue="cluster", kind="hist", rug=True
# )

pivot_df = df.pivot_table(index="cluster", columns="DPGRSUM", aggfunc="sum")["WEIGHT"]
pivot_df / pivot_df.sum() * 100

# %% [markdown]
# Dimensionality Reduction Models

# %%
# E1: ML Dimensionality Reduction
