import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

from pandas.api.types import infer_dtype


def recode_column(series, recode_dict):
    series = series.copy()
    for key, value in recode_dict.items():
        # print(f"recoding {key} to {value}")
        series.loc[series == key] = value
    best_dtype = infer_dtype(series, skipna=True)
    if "mixed" in best_dtype:
        # use the last type in mixed type like 'mixed-integer-float'
        best_type = best_dtype.split("-")[-1]
        if best_dtype == "mixed":
            # no numerics for mixed, set to str
            best_dtype = "str"
    elif best_dtype == "floating":
        best_dtype = "float"
    else:
        best_dtype = "str"
    return series.astype(best_dtype)


class pumf_2021_util:

    @staticmethod
    def read_indiv_table():

        preloaded_fea_file = "data/census_pumf_vancouver_2021.fea"
        data_path = (
            "../data_packages/PUMF_Census_2021_StatCan/ind/data_donnees_2021_ind_v2.csv"
        )
        # data_path = "../data_packages/PUMF_Census_2021_StatCan/hier/data_donnees_2021_hier.csv"

        if not os.path.exists(preloaded_fea_file):
            df = pd.read_csv(data_path)
            # filter for CMA equals to Vancouver
            # source: https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=317043&CVD=317046&CPV=59A&CST=01012016&CLV=3&MLV=5
            df = df[df["CMA"] == 933].reset_index(
                drop=True
            )  # filter by Vancouver CMA only
            # save to feather to allow faster loading later
            df.to_feather(preloaded_fea_file)
        else:
            df = pd.read_feather(preloaded_fea_file)

        return df

    @staticmethod
    def recode_indiv_table(df):

        col_recode = {
            "Gender": {
                1: "Woman+",
                2: "Man+",
            },
            "AGEGRP": {
                1: "00 to 09",
                2: "00 to 09",
                3: "00 to 09",
                4: "10 to 17",
                5: "10 to 17",
                6: "10 to 17",
                7: "18 to 24",
                8: "18 to 24",
                9: "25 to 34",
                10: "25 to 34",
                11: "35 to 44",
                12: "35 to 44",
                13: "45 to 54",
                14: "45 to 54",
                15: "55 to 64",
                16: "55 to 64",
                17: "65 to 74",
                18: "65 to 74",
                19: "75 or older",
                20: "75 or older",
                21: "75 or older",
                88: "Not available",
            },
            "HHSIZE": {
                1: "1 person",
                2: "2 persons",
                3: "3 persons",
                4: "4 persons",
                5: "5 persons",
                6: "6 persons",
                7: "7+ persons",
                8: "Not available",
            },
            "HHSIZE->HHSIZE4": {
                "1 person": "1 person",
                "2 persons": "2 persons",
                "3 persons": "3 persons",
                "4 persons": "4+ persons",
                "5 persons": "4+ persons",
                "6 persons": "4+ persons",
                "7+ persons": "4+ persons",
            },
            # add education - highest degrees
            "HDGREE->HighestDegrees": {
                1: "1 - HS or No Degree",
                2: "1 - HS or No Degree",
                3: "2 - Diplomas or Certificates",
                4: "2 - Diplomas or Certificates",
                5: "2 - Diplomas or Certificates",
                6: "2 - Diplomas or Certificates",
                7: "2 - Diplomas or Certificates",
                8: "2 - Diplomas or Certificates",
                9: "3 - Bachelor",
                10: "3 - Bachelor",
                11: "5 - Doctorates",  # MD
                12: "4 - Masters",
                13: "5 - Doctorates",
                88: "Not available",
                99: "Not applicable",
            },
            # add NOC21
            "NOC21->NOCS_cat": {
                1: "0 Management",
                2: "0 Management",
                3: "1 Business, finance, admin",
                4: "1 Business, finance, admin",
                5: "1 Business, finance, admin",
                6: "1 Business, finance, admin",
                7: "2 - Natural and applied sciences",
                8: "2 - Natural and applied sciences",
                9: "3 - Health",
                10: "3 - Health",
                11: "3 - Health",
                12: "4 - education, law and social, community and government",
                13: "4 - education, law and social, community and government",
                14: "4 - education, law and social, community and government",
                15: "5 - art, culture, recreation and sport",
                16: "5 - art, culture, recreation and sport",
                17: "6 - Sales and service",
                18: "6 - Sales and service",
                19: "6 - Sales and service",
                20: "6 - Sales and service",
                21: "7 - Trades, transport and equipment operators",
                22: "7 - Trades, transport and equipment operators",
                23: "7 - Trades, transport and equipment operators",
                24: "7 - Trades, transport and equipment operators",
                25: "8 - Natural resources, agriculture",
                26: "9 - manufacturing and utilities",
                88: "Not available",
                99: "Not applicable",
            },
            # add NOC21
            "NOC21->NOC21_cat": {
                1: "00 Legislative and senior managers",
                2: "0x Middle management occupations",
                3: "11 Professional occupations in business and finance",
                4: "12 Administrative and financial supervisors",
                5: "13 Administrative occupations and transportation logistics occupations",
                6: "14 Administrative and financial support and supply chain logistics occupations",
                7: "21 Professional occupations in natural and applied sciences",
                8: "22 Technical occupations related to natural and applied sciences",
                9: "31 Professional occupations in health",
                10: "32 Technical occupations in health ",
                11: "33 Assisting occupations in support of health services",
                12: "41 Professional occupations in law, education, social, community and government services",
                13: "42 Front-line public protection services and paraprofessional ",
                14: "43-45 Assisting occupations, care providers",
                15: "51-52 Professional and technical occupations",
                16: "53-55 Other occupations in art, culture and sport",
                17: "62 Retail sales and service supervisors",
                18: "63 Occupations in sales and services",
                19: "64 Sales and service representatives",
                20: "65 Sales and service support",
                21: "72 Technical trades and transportation officers and controllers",
                22: "73 General trades",
                23: "74 Mail and message distribution, other transport equipment",
                24: "75 Helpers and labourers",
                25: "82-85 Occupations in natural resources",
                26: "92-95 Occupations in processing, manufacturing and utilities",
                88: "Not available",
                99: "Not applicable",
            },
            "WKSWRK": {
                # 0: "00 did not work in 2020",
                # 1: "01 to 09 weeks",
                # 2: "10 to 19 weeks",
                # 3: "20 to 29 weeks",
                # 4: "30 to 39 weeks",
                # 5: "40 to 48 weeks",
                # 6: "49 to 52 weeks",
                0: "09 weeks or less",
                1: "09 weeks or less",
                2: "10 to 19 weeks",
                3: "20 to 29 weeks",
                4: "30 to 39 weeks",
                5: "40 to 48 weeks",
                6: "49 to 52 weeks",
                8: "Not available",
                9: "Not applicable",
            },
            "WRKACT": {
                1: "Did not work in 2020",
                2: "Did not work in 2020",
                3: "full time - 01 to 13 weeks",
                4: "part time - 01 to 13 weeks",
                5: "full time - 14 to 26 weeks",
                6: "part time - 14 to 26 weeks",
                7: "full time - 27 to 39 weeks",
                8: "part time - 27 to 39 weeks",
                9: "full time - 40 to 48 weeks",
                10: "part time - 40 to 48 weeks",
                11: "full time - 49 to 52 weeks",
                12: "part time - 49 to 52 weeks",
                88: "Not available",
                99: "Not applicable",
            },
            "FPTWK": {
                1: "full time",
                2: "part time",
                9: "Not applicable",
            },
            "VISMIN": {
                1: "Not Visiable Minority",
                2: "Is Visible Minority",
                3: "Is Visible Minority",
                4: "Is Visible Minority",
                5: "Is Visible Minority",
                6: "Is Visible Minority",
                7: "Is Visible Minority",
                8: "Is Visible Minority",
                9: "Is Visible Minority",
                10: "Is Visible Minority",
                11: "Is Visible Minority",
                12: "Is Visible Minority",
                13: "Is Visible Minority",
                88: "Not available",
            },
            # FIXME: PKID0_1 + PKID2_5 + PKID6_14 + PKID15_24 to calclate dependent kids
            "PKIDS": {
                0: "No kids",
                1: "One or more kids",
                8: "Not available",
                9: "Not applicable",
            },
            "MarStH->Marital_Status": {
                1: "Single",
                2: "Married_CL",
                3: "Married_CL",
                4: "Separated",
                5: "Separated",
                6: "Separated",
                8: "Not available",
            },
            "MarStH->Marital": {
                1: "Single_or_Separated",
                2: "Married_CL",
                3: "Married_CL",
                4: "Single_or_Separated",
                5: "Single_or_Separated",
                6: "Single_or_Separated",
                8: "Not available",
            },
            "CFInc": {
                1: 1000,
                2: 3000,
                2: 3500,
                3: 6000,
                4: 8500,
                5: 11000,
                6: 13500,
                7: 16000,
                8: 18500,
                9: 22500,
                10: 27500,
                11: 32500,
                12: 37500,
                13: 42500,
                14: 47500,
                15: 52500,
                16: 57500,
                17: 62500,
                18: 67500,
                19: 72500,
                20: 77500,
                21: 82500,
                22: 87500,
                23: 92500,
                24: 97500,
                25: 105000,
                26: 115000,
                27: 125000,
                28: 135000,
                29: 145000,
                30: 162500,
                31: 187500,
                32: 225000,
                33: 250000,
            },
        }

        # data processing of all existing variables
        for col, rd in col_recode.items():
            if "->" in col:
                old_col = col.split("->")[0]
                new_col = col.split("->")[1]
                df[new_col] = recode_column(series=df[old_col], recode_dict=rd)
            else:
                df[col] = recode_column(series=df[col], recode_dict=rd)

        # # Employment type
        # for cat in list(df.NOCS.unique()):
        #     nocs_cat = cat.split(" ")[0]
        #     # nocs_cat = cat[0:6]
        #     df.loc[df.NOCS == cat, "NOCS_cat"] = nocs_cat
        # # Combine employment categories
        # df["NOCS_cat_combined"] = df.NOCS_cat.replace(["A", "C"], "mgmt_sci")
        # df["NOCS_cat_combined"] = df.NOCS_cat.replace(["B"], "busi")
        # df["NOCS_cat_combined"] = df.NOCS_cat.replace(["D", "E"], "health_edu_law")
        # df["NOCS_cat_combined"] = df.NOCS_cat.replace(
        #     ["F", "H", "J"], "art_trades_manuf"
        # )
        # df["NOCS_cat_combined"] = df.NOCS_cat.replace(["G", "I"], "sales_agri")

        # # encode variables
        # # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features
        # # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder
        # lab_encode = LabelEncoder()
        # df["Sex_cat"] = lab_encode.fit_transform(df.Sex)
        # categories = list(lab_encode.classes_)
        # print(dict(zip(categories, lab_encode.transform(categories))))
