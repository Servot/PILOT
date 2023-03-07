# %%
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.impute import KNNImputer
from io import BytesIO
from zipfile import ZipFile
import requests
from pathlib import Path

DATAPATH = Path().absolute() / "Data"

# %%
# temperature: categorical=np.array([0])
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00514/Bias_correction_ucl.csv"
temp = pd.read_csv(path)
# feature transformation
temp = temp.dropna()
temp["Date"] = pd.to_datetime(temp["Date"])
temp["Day"] = temp["Date"].dt.weekday
temp["Month"] = temp["Date"].dt.month
temp["Year"] = temp["Date"].dt.year
temp.drop("Date", axis=1, inplace=True)
yr = {2013: 1, 2014: 2, 2015: 3, 2016: 4, 2017: 5}
temp["Year"] = temp["Year"].map(yr)
temp.station = temp.station.astype(int)
temp.drop(["Next_Tmin"], axis=1).rename(columns={"Next_Tmax": "target"}).to_csv(
    DATAPATH / "Bias_correction_ucl.csv", index=False
)

# %%
# communities
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
cac = pd.read_csv(path, sep=",", header=None, na_values="?")
# feature transformation
cac = cac.drop([0, 1, 2, 3, 4], axis=1)
for i in range(5, 128):
    if cac[i].isnull().any():
        cac[i].fillna(cac[i].mean(), inplace=True)
cac.iloc[:, -1] = cac.iloc[:, -1].astype(float)
cac.rename(columns={cac.columns[-1]: "target"}).to_csv(DATAPATH / "communities.csv", index=False)

# %%
# riboflavin
ribo = pd.read_csv(DATAPATH / "ribo.csv")
ribo.drop(columns="Unnamed: 0").rename(columns={ribo.columns[-1]: "target"}).to_csv(
    DATAPATH / "ribo_preprocessed.csv", index=False
)


# %%
# california housing
housing = datasets.fetch_california_housing(as_frame=True)
housing.frame.rename(columns={housing.target_names[0]: "target"}).to_csv(DATAPATH / "housing.csv")

# %%
# bike
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"
bike = pd.read_csv(path, encoding="unicode_escape")
# feature transformation
date = bike.Date.unique()
date_encoder = {day: i for i, day in enumerate(date)}
bike.Holiday = bike.Holiday.map({"No Holiday": 0, "Holiday": 1})
bike.iloc[:, -1] = bike.iloc[:, -1].map({"Yes": 1, "No": 0})
bike.Date = bike.Date.map(date_encoder)
bike.Date = bike.Date % 7
bike.Seasons = bike.Seasons.map({"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3})
bike.Holiday = bike.Holiday.astype(int)
bike.iloc[:, -1] = bike.iloc[:, -1].astype(int)
bike.iloc[:, 1] = bike.iloc[:, 1].astype(float)
bike.rename(columns={bike.columns[1]: "target"}).to_csv(DATAPATH / "SeoulBikeData.csv", index=False)


# %%
# walmart: categorical=np.array([0])
wal = pd.read_csv(DATAPATH / "Walmart.csv")
# feature transformation
wal["Date"] = pd.to_datetime(wal["Date"], dayfirst=True)
wal["Day"] = wal["Date"].dt.weekday
wal["Month"] = wal["Date"].dt.month
wal["Year"] = wal["Date"].dt.year
wal.drop("Date", axis=1, inplace=True)
yr = {2010: 1, 2011: 2, 2012: 3}
wal["Year"] = wal["Year"].map(yr)
wal.rename(columns={"Weekly_Sales": "target"}).to_csv(
    DATAPATH / "Walmart_preprocessed.csv", index=False
)

# %%
# electricity
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv"
elec = pd.read_csv(path)
elec = elec.drop(["p1", "stabf"], axis=1)

elec.rename(columns={elec.columns[-1]: "target"}).to_csv(DATAPATH / "electricity.csv", index=False)

# %%
# diabetes
diabetes = datasets.load_diabetes(as_frame=True)
# target is already called 'target'
diabetes.frame.to_csv(DATAPATH / "diabetes.csv")

# %%
# airfoil
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
air = pd.read_table(path, header=None)
air.rename(columns={air.columns[-1]: "target"}).to_csv(DATAPATH / "airfoil.csv", index=False)


# %%
# wine
path = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
)
wine = pd.read_csv(path, sep=";")
wine.iloc[:, -1] = wine.iloc[:, -1].astype(float)
wine.rename(columns={"quality": "target"}).to_csv(DATAPATH / "wine.csv", index=False)


# %%
# boston
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None, engine="python")
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, [2]].astype(float)

pd.DataFrame(np.hstack([data, target]), columns=list(range(data.shape[-1])) + ["target"]).to_csv(
    DATAPATH / "boston.csv", index=False
)

# %%
# abalone: categorical=np.array([0])
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
abalone = pd.read_table(path, sep=",", header=None)
abalone[0] = abalone[0].map({"M": 0, "F": 1, "I": 2})

abalone.iloc[:, -1] = abalone.iloc[:, -1].astype(float)
abalone.rename(columns={abalone.columns[-1]: "target"}).to_csv(
    DATAPATH / "abalone.csv", index=False
)

# %%
# skills
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00272/SkillCraft1_Dataset.csv"
skills = pd.read_csv(path, na_values="?")
# feature transformation
skills = skills.drop(["GameID"], axis=1)
skills = skills.dropna()
skills.loc[skills.TotalHours >= 3000, "TotalHours"] = 3000
skills.iloc[:, 0] = skills.iloc[:, 0].astype(float)
skills.rename(columns={skills.columns[0]: "target"}).to_csv(DATAPATH / "skills.csv", index=False)


# %%
# ozone: categorical=np.array([10, 11])
ozone_pd = pd.read_csv(DATAPATH / "ozone.csv")
ozone_x = ozone_pd.iloc[:, 1:]
# categorical variables
vent = {"Nord": 0, "Est": 1, "Quest": 2, "Sud": 3}
pluie = {"Sec": 0, "Pluie": 1}
ozone_x["vent"] = ozone_x["vent"].map(vent)
ozone_x["pluie"] = ozone_x["pluie"].map(pluie)
ozone_x = ozone_x.fillna(-20000)
# imputer missing predictor values
knnipt = KNNImputer(missing_values=-20000)
ozone_x = knnipt.fit_transform(ozone_x)
# adjust for ordinal/categorical variables
ozone_x[:, -1] = np.round(ozone_x[:, -1])
ozone_x[:, -2] = np.round(ozone_x[:, -2])
ozone_x[:, -3] = np.round(ozone_x[:, -3])
ozone_x[:, 3] = np.round(ozone_x[:, 3])
ozone_x[:, 4] = np.round(ozone_x[:, 4])
ozone_x[:, 5] = np.round(ozone_x[:, 5])
# impute responses
ozone_pd.iloc[:, 1:] = ozone_x
knnipt = KNNImputer()
ozone_data = knnipt.fit_transform(ozone_pd)
pd.DataFrame(ozone_data, columns=["target"] + list(range(ozone_data.shape[-1] - 1))).to_csv(
    DATAPATH / "ozone_preprocessed.csv", index=False
)

# %%
# concrete
#!pip install --upgrade xlrd
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
con = pd.read_excel(path)

con.rename(columns={con.columns[-1]: "target"}).to_csv(DATAPATH / "concrete.csv", index=False)

# %%
# energy
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
energy = pd.read_excel(path, engine="openpyxl")
energy.drop(columns=["Y1"]).rename(columns={"Y2": "target"}).to_csv(
    DATAPATH / "energy.csv", index=False
)


# %%
# residential
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00437/Residential-Building-Data-Set.xlsx"
res = pd.read_excel(path, engine="openpyxl", header=1)
res.loc[:, "V-10"] = res.loc[:, "V-10"].astype(float)
res.drop(columns=["V-9"]).rename(columns={"V-10": "target"}).to_csv(
    DATAPATH / "residential.csv", index=False
)


# %%
# superconductor
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip"
zip_file = ZipFile(BytesIO(requests.get(url).content))
trainsc = pd.read_csv(zip_file.open(zip_file.infolist()[1]))
trainsc = trainsc.drop_duplicates()

trainsc.rename(columns={trainsc.columns[-1]: "target"}).to_csv(
    DATAPATH / "superconductor.csv", index=False
)


# %%
# bodyfat
path = DATAPATH / "bodyfat.csv"
bf = pd.read_csv(path)

bf.rename(columns={"Wrist": "target"}).to_csv(DATAPATH / "bodyfat_preprocessed.csv", index=False)

# %%
# graduate admission
path = DATAPATH / "ga.csv"
ga = pd.read_csv(path)
ga.drop("Serial No.", axis=1, inplace=True)
ga.rename(columns={"Chance of Admit ": "target"}).to_csv(
    DATAPATH / "ga_preprocessed.csv", index=False
)

# %%
