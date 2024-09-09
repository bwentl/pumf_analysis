import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler


class pumf_2016_util:

    @staticmethod
    def read_indiv_table():

        preloaded_fea_file = "data/census_pumf_vancouver_2016.fea"
        data_path = "../data_packages/PUMF_Census_2016/dataverse_files/Data/Census_2016_Individual_PUMF.sav"

        if not os.path.exists(preloaded_fea_file):
            df = pd.read_spss(data_path)
            # filter for CMA equals to Vancouver
            # source: https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=317043&CVD=317046&CPV=59A&CST=01012016&CLV=3&MLV=5
            df = df[df["CMA"] == "Vancouver"].reset_index(
                drop=True
            )  # filter by Vancouver CMA only
            # save to feather to allow faster loading later
            df.to_feather(preloaded_fea_file)
        else:
            df = pd.read_feather(preloaded_fea_file)

        return df

    @staticmethod
    def recode_indiv_table(df):

        # data processing of age_group
        for age_group in list(df.AGEGRP.unique()):
            ag_str = age_group.replace(" years", "")
            if ag_str == "Not available":
                lower = -1
                upper = -1
            elif ag_str == "85 and over":
                lower = 85
                upper = 85
            else:
                lower = int(ag_str.split(" to ")[0])
                upper = int(ag_str.split(" to ")[1])
            df.loc[df.AGEGRP == age_group, "AGE"] = (lower + upper) / 2
        bins = list(range(0, 80, 5)) + [99]
        df["AGEBIN"] = pd.cut(df["AGE"], bins)

        # recode variables
        # Moved recently (1 year)
        for cat in list(df.MOB1.unique()):
            if (cat == "Non-movers") or (cat == "Non-migrants"):
                df.loc[df.MOB1 == cat, "MOB1_cat"] = "not_moved_1"
            elif (
                (cat == "Interprovincial migrants")
                or (cat == "Different CSD, same census division")
                or (cat == "Different CD, same province")
            ):
                df.loc[df.MOB1 == cat, "MOB1_cat"] = "moved_within_canada_1"
            elif cat == "External migrants":
                df.loc[df.MOB1 == cat, "MOB1_cat"] = "moved_to_canada_1"
        # Moved within 5 years
        for cat in list(df.Mob5.unique()):
            if (cat == "Non-movers") or (cat == "Non-migrants"):
                df.loc[df.Mob5 == cat, "MOB5_cat"] = "not_moved_5"
            elif (
                (cat == "Interprovincial migrants")
                or (cat == "Different CSD, same census division")
                or (cat == "Different CD, same province")
            ):
                df.loc[df.Mob5 == cat, "MOB5_cat"] = "moved_within_canada_5"
            elif cat == "External migrants":
                df.loc[df.Mob5 == cat, "MOB5_cat"] = "moved_to_canada_5"
        # Household size
        for cat in list(df.HHSIZE.unique()):
            if cat == "Not available":
                df.loc[df.HHSIZE == cat, "HHSIZE_int"] = 1
            else:
                hh_size = cat.split(" ")[0]
                df.loc[df.HHSIZE == cat, "HHSIZE_int"] = hh_size
        # Employment type
        for cat in list(df.NOCS.unique()):
            nocs_cat = cat.split(" ")[0]
            # nocs_cat = cat[0:6]
            df.loc[df.NOCS == cat, "NOCS_cat"] = nocs_cat
        # Combine employment categories
        df["NOCS_cat_combined"] = df.NOCS_cat.replace(["A", "C"], "mgmt_sci")
        df["NOCS_cat_combined"] = df.NOCS_cat.replace(["B"], "busi")
        df["NOCS_cat_combined"] = df.NOCS_cat.replace(["D", "E"], "health_edu_law")
        df["NOCS_cat_combined"] = df.NOCS_cat.replace(
            ["F", "H", "J"], "art_trades_manuf"
        )
        df["NOCS_cat_combined"] = df.NOCS_cat.replace(["G", "I"], "sales_agri")

        # encode variables
        # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder
        lab_encode = LabelEncoder()
        df["Sex_cat"] = lab_encode.fit_transform(df.Sex)
        categories = list(lab_encode.classes_)
        print(dict(zip(categories, lab_encode.transform(categories))))

        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder
        cats = [
            [
                "No bedroom",
                "1 bedroom",
                "2 bedrooms",
                "3 bedrooms",
                "4 bedrooms",
                "5 bedrooms or more",
                "Not available",
            ]
        ]
        ord_encode = OrdinalEncoder()
        ord_encode = ord_encode.set_params(encoded_missing_value=-1, categories=cats)
        df["BedRm_cat"] = ord_encode.fit_transform(df.BedRm.to_numpy().reshape(-1, 1))
        print(
            dict(
                zip(
                    cats[0],
                    ord_encode.transform(np.array(cats).reshape(-1, 1)).reshape(1, -1)[
                        0
                    ],
                )
            )
        )
