# %%
import numpy as np
import pandas as pd


def get_additional_stats(raw_file_path: str) -> pd.DataFrame:
    # ## Preisindex
    preis_index = pd.read_excel(
        f"{raw_file_path}/additional_stats/hauspreis_index.xlsx"
    )
    preis_index = preis_index.iloc[2:5, 2:]
    preis_index = preis_index.T
    preis_index = preis_index.reset_index(drop=True)
    preis_index.columns = ["year", "quartal", "house_price_index"]
    preis_index["year"] = preis_index["year"].ffill()
    preis_index["quartal"] = preis_index["quartal"].apply(lambda x: int(x[0]))
    preis_index["year"] = preis_index["year"].apply(lambda x: int(x))
    preis_index["house_price_index"] = preis_index["house_price_index"].apply(
        lambda x: float(x) if x != "..." else np.nan
    )
    preis_index["house_price_index"] = preis_index["house_price_index"].ffill()

    # %% md
    # ### Baugewerbe
    # %%
    def save_cast(value, dtype):
        try:
            return dtype(value)
        except:
            return np.nan

    baugewerbe = pd.read_excel(f"{raw_file_path}/additional_stats/baugewerbe.xlsx")
    baugewerbe = baugewerbe.iloc[5:-3, :]
    baugewerbe.columns = [
        "year",
        "month",
        "construction_industry_employees",
        "construction_industry_employees_change",
        "construction_industry_turnover",
        "construction_industry_turnover_change",
    ]
    baugewerbe["year"] = baugewerbe["year"].ffill()
    baugewerbe["year"] = baugewerbe["year"].apply(lambda x: int(x))
    baugewerbe["construction_industry_employees"] = baugewerbe[
        "construction_industry_employees"
    ].apply(lambda x: int(x) if x != "..." else np.nan)
    baugewerbe["construction_industry_turnover"] = baugewerbe[
        "construction_industry_turnover"
    ].apply(lambda x: float(x) if x != "..." else np.nan)
    baugewerbe["construction_industry_employees_change"] = baugewerbe[
        "construction_industry_employees_change"
    ].apply(lambda x: save_cast(x, float))
    baugewerbe["construction_industry_turnover_change"] = baugewerbe[
        "construction_industry_turnover_change"
    ].apply(lambda x: save_cast(x, float))
    baugewerbe["construction_industry_employees"] = baugewerbe[
        "construction_industry_employees"
    ].ffill()
    baugewerbe["construction_industry_turnover"] = baugewerbe[
        "construction_industry_turnover"
    ].ffill()
    baugewerbe["month"] = baugewerbe["month"].apply(lambda x: x.strip())
    # Map Januar to 1, Februar to 2, etc.
    baugewerbe["month"] = baugewerbe["month"].map(
        {
            "Januar": 1,
            "Februar": 2,
            "März": 3,
            "April": 4,
            "Mai": 5,
            "Juni": 6,
            "Juli": 7,
            "August": 8,
            "September": 9,
            "Oktober": 10,
            "November": 11,
            "Dezember": 12,
        }
    )
    baugewerbe["construction_industry_turnover_per_employee"] = (
        baugewerbe["construction_industry_turnover"]
        / baugewerbe["construction_industry_employees"]
    )

    # %% md
    # ## Arbeitsmarkt
    # %%
    arbeitsmarkt = pd.read_excel(
        f"{raw_file_path}/additional_stats/arbeitslosenquote.xlsx"
    )
    arbeitsmarkt = arbeitsmarkt.iloc[5:-7, :10]
    arbeitsmarkt = arbeitsmarkt.drop(
        columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4", "Unnamed: 5", "Unnamed: 6"]
    )
    arbeitsmarkt.columns = [
        "year",
        "month",
        "employed_mio",
        "unemployed_mio",
        "unemployed_rate",
    ]
    arbeitsmarkt["year"] = arbeitsmarkt["year"].ffill()
    arbeitsmarkt["year"] = arbeitsmarkt["year"].apply(lambda x: int(x))
    arbeitsmarkt["month"] = arbeitsmarkt["month"].apply(lambda x: x.strip())
    arbeitsmarkt["unemployed_rate"] = arbeitsmarkt["unemployed_rate"].apply(
        lambda x: save_cast(x, float)
    )
    arbeitsmarkt["unemployed_rate"] = arbeitsmarkt["unemployed_rate"].ffill()
    arbeitsmarkt["employed_mio"] = arbeitsmarkt["employed_mio"].apply(
        lambda x: save_cast(x, float)
    )
    arbeitsmarkt["employed_mio"] = arbeitsmarkt["employed_mio"].ffill()
    arbeitsmarkt["unemployed_mio"] = arbeitsmarkt["unemployed_mio"].apply(
        lambda x: save_cast(x, float)
    )
    arbeitsmarkt["unemployed_mio"] = arbeitsmarkt["unemployed_mio"].ffill()
    arbeitsmarkt["month"] = arbeitsmarkt["month"].map(
        {
            "Januar": 1,
            "Februar": 2,
            "März": 3,
            "April": 4,
            "Mai": 5,
            "Juni": 6,
            "Juli": 7,
            "August": 8,
            "September": 9,
            "Oktober": 10,
            "November": 11,
            "Dezember": 12,
        }
    )

    # ## Lohnindex
    lohnindex = pd.read_excel(f"{raw_file_path}/additional_stats/lohnindex.xlsx")
    lohnindex = lohnindex.iloc[4:-3, :]
    lohnindex.columns = [
        "year",
        "quartal",
        "reallohnindex",
        "reallohnindex_change",
        "nominallohnindex",
        "nominallohnindex_change",
    ]
    lohnindex = lohnindex.drop(
        columns=["reallohnindex_change", "nominallohnindex_change"]
    )
    lohnindex["year"] = lohnindex["year"].ffill()
    lohnindex["year"] = lohnindex["year"].apply(lambda x: int(x))
    lohnindex["reallohnindex"] = lohnindex["reallohnindex"].apply(
        lambda x: float(x) if x != "..." else np.nan
    )
    lohnindex["nominallohnindex"] = lohnindex["nominallohnindex"].apply(
        lambda x: float(x) if x != "..." else np.nan
    )
    lohnindex["reallohnindex"] = lohnindex["reallohnindex"].ffill()
    lohnindex["nominallohnindex"] = lohnindex["nominallohnindex"].ffill()
    lohnindex["quartal"] = lohnindex["quartal"].apply(lambda x: int(x[0]))

    # ## Wirtschaftswachtum
    wirtschaftswachtum = pd.read_excel(
        f"{raw_file_path}/additional_stats/wertschoepfung.xlsx"
    )
    wirtschaftswachtum = wirtschaftswachtum.iloc[:11, :][2:].T
    # Remove first row
    wirtschaftswachtum = wirtschaftswachtum[2:]
    wirtschaftswachtum.columns = [
        "year",
        "empty",
        "gross_value_creation",
        "bruttowertschoepfung_ohne_guetersubventionen",
        "bruttonwertschoepfung_guetersteuern",
        "bruttowertschoepfung_guetersubventionen",
        "gpd",
        "gdp_change",
        "gdp_per_citizen",
    ]
    wirtschaftswachtum = wirtschaftswachtum.drop(
        columns=[
            "bruttowertschoepfung_ohne_guetersubventionen",
            "bruttonwertschoepfung_guetersteuern",
            "bruttowertschoepfung_guetersubventionen",
            "empty",
        ]
    ).reset_index(drop=True)
    for col in wirtschaftswachtum.columns:
        if col == "year":
            wirtschaftswachtum[col] = wirtschaftswachtum[col].apply(lambda x: int(x))
        else:
            wirtschaftswachtum[col] = wirtschaftswachtum[col].apply(
                lambda x: save_cast(x, float)
            )

    # ## Baulandpreise
    baulandpreise = pd.read_csv(f"{raw_file_path}/additional_stats/baulandpreise.csv")
    baulandpreise["year"] = baulandpreise["Jahr"].apply(
        lambda x: int(x[-2:]) + 2000,
    )
    # Map Jan, Feb, etc. to 1, 2, etc.
    baulandpreise["month"] = (
        baulandpreise["Jahr"]
        .apply(lambda x: x[:3])
        .map(
            {
                "Jan": 1,
                "Feb": 2,
                "Mar": 3,
                "Apr": 4,
                "May": 5,
                "Jun": 6,
                "Jul": 7,
                "Aug": 8,
                "Sep": 9,
                "Oct": 10,
                "Nov": 11,
                "Dez": 12,
                "Dec": 12,
            }
        )
    )
    baulandpreise["month"] = baulandpreise["month"].astype(int)
    baulandpreise = baulandpreise.drop(columns=["Jahr"])
    baulandpreise.columns = [
        "build_land_price_index_flat",
        "build_land_price_index_new_constructed",
        "build_land_price_index_existing_buildings",
        "build_land_price_index_total",
        "year",
        "month",
    ]

    # ## Konsumbereitschaft
    konsumbereitschaft = pd.read_excel(
        f"{raw_file_path}/additional_stats/konsumbereitschaft.xlsx"
    )
    konsumbereitschaft = konsumbereitschaft.iloc[7:, 2:]
    for col in konsumbereitschaft.columns:
        konsumbereitschaft[col] = konsumbereitschaft[col].apply(
            lambda x: save_cast(x, float)
        )
    konsumbereitschaft = konsumbereitschaft.sum()
    # Apply with index
    quartal = [1, 2, 3, 4] * (len(konsumbereitschaft) // 4)
    year = [[year] * 4 for year in range(1991, 2025)]
    year = [item for sublist in year for item in sublist]
    konsumbereitschaft = konsumbereitschaft.reset_index(drop=True)
    konsumbereitschaft = pd.DataFrame(
        {"year": year, "quartal": quartal, "consumption_index": konsumbereitschaft}
    )
    konsumbereitschaft["consumption_index"] = konsumbereitschaft[
        "consumption_index"
    ].replace(0, np.nan)
    konsumbereitschaft["consumption_index"] = konsumbereitschaft[
        "consumption_index"
    ].ffill()

    # ## Zinssätze

    zinssaetze = pd.read_csv(
        f"{raw_file_path}/additional_stats/wohnungsbauzinsen.csv", sep=";"
    )
    zinssaetze = zinssaetze.iloc[8:-1, :2]
    zinssaetze.columns = ["date", "housing_interest_rate"]

    zinssaetze["year"] = zinssaetze["date"].apply(
        lambda x: save_cast(str(x).split("-")[0], int)
    )
    zinssaetze["month"] = zinssaetze["date"].apply(
        lambda x: save_cast(str(x).split("-")[1], int)
    )
    zinssaetze["housing_interest_rate"] = zinssaetze["housing_interest_rate"].apply(
        lambda x: save_cast(x.replace(",", "."), float)
    )
    zinssaetze = zinssaetze.drop(columns=["date"])

    # ## Bevölkerungsentwicklung
    bevoelkerungsentwicklung = pd.read_excel(
        f"{raw_file_path}/additional_stats/bevoelkerungsentwicklung.xlsx"
    )
    bevoelkerungsentwicklung = bevoelkerungsentwicklung.iloc[4:-5, :]
    bevoelkerungsentwicklung.columns = ["year", "population"]
    bevoelkerungsentwicklung["year"] = bevoelkerungsentwicklung["year"].apply(
        lambda x: int(x[-4:])
    )
    bevoelkerungsentwicklung["population"] = bevoelkerungsentwicklung[
        "population"
    ].apply(lambda x: int(x))

    # ## Alterstruktur
    alterstruktur = pd.read_csv(
        f"{raw_file_path}/additional_stats/15_bevoelkerungsvorausberechnung_daten.csv",
        sep=";",
    )
    alterstruktur = alterstruktur.groupby("Simulationsjahr").sum()
    alterstruktur = alterstruktur.drop(columns=["Variante", "mw"])

    alterstruktur = alterstruktur.T

    series = {}
    alterstruktur_under_25 = alterstruktur.loc[
        [f"Bev_{i}_{i + 1}" for i in range(0, 25)]
    ].sum()
    series["Pop_under_25"] = alterstruktur_under_25
    for i in range(25, 60, 5):
        alterstruktur_5 = alterstruktur.loc[
            [f"Bev_{i}_{i + 1}" for i in range(i, i + 5)]
        ].sum()
        series[f"Pop_{i}_{i + 5}"] = alterstruktur_5

    alterstruktur_over_60 = alterstruktur.loc[
        [f"Bev_{i}_{i + 1}" for i in range(60, 100)]
    ].sum()

    series["Pop_over_60"] = alterstruktur_over_60
    alterstruktur = pd.DataFrame(series)

    # Divide by total population
    alterstruktur["Bev"] = alterstruktur.sum(axis=1)
    for col in alterstruktur.columns:
        if "Pop" in col:
            alterstruktur[col] = alterstruktur[col] / alterstruktur["Bev"]

    alterstruktur = alterstruktur.drop(columns=["Bev"])
    # Rename Simulationsjahr to year
    alterstruktur.index = alterstruktur.index.rename("year")
    alterstruktur = alterstruktur.reset_index()

    ## Baugenehmigungen
    baugenehmigungen = pd.read_excel(
        f"{raw_file_path}/additional_stats/baugenehmigungen.xlsx"
    )
    baugenehmigungen = baugenehmigungen.iloc[7:-9, [1, 26, 34]]
    baugenehmigungen.columns = ["year", "num_new_buildings", "num_new_flats"]
    baugenehmigungen["year"] = baugenehmigungen["year"].apply(lambda x: int(x))
    baugenehmigungen["num_new_buildings"] = baugenehmigungen["num_new_buildings"].apply(
        lambda x: save_cast(x, int)
    )
    baugenehmigungen["num_new_flats"] = baugenehmigungen["num_new_flats"].apply(
        lambda x: save_cast(x, int)
    )

    # ## Baulandpreise
    baulandpreise = pd.read_csv(f"{raw_file_path}/additional_stats/baulandpreise.csv")
    baulandpreise["year"] = baulandpreise["Jahr"].apply(lambda x: int(x[-2:]) + 2000)
    baulandpreise["month"] = (
        baulandpreise["Jahr"]
        .apply(lambda x: x[:3])
        .map(
            {
                "Jan": 1,
                "Feb": 2,
                "Mar": 3,
                "Apr": 4,
                "May": 5,
                "Jun": 6,
                "Jul": 7,
                "Aug": 8,
                "Sep": 9,
                "Oct": 10,
                "Nov": 11,
                "Dec": 12,
                "Dez": 12,
            }
        )
    )
    baulandpreise["month"] = baulandpreise["month"].astype(int)
    baulandpreise = baulandpreise.drop(columns=["Jahr"])
    baulandpreise.columns = [
        "build_land_price_index_flat",
        "build_land_price_index_new_constructed",
        "build_land_price_index_existing_buildings",
        "build_land_price_index_total",
        "year",
        "month",
    ]

    frames = [
        preis_index,
        lohnindex,
        arbeitsmarkt,
        baugewerbe,
        wirtschaftswachtum,
        baulandpreise,
        konsumbereitschaft,
        zinssaetze,
        bevoelkerungsentwicklung,
        alterstruktur,
        baugenehmigungen,
        baulandpreise,
    ]
    for frame in frames:
        if not "quartal" in frame.columns and "month" in frame.columns:
            frame["quartal"] = frame["month"].apply(lambda x: (x - 1) // 3 + 1)

    frames_with_month = [frame for frame in frames if "month" in frame.columns]
    frames_with_quartal = [
        frame
        for frame in frames
        if "quartal" in frame.columns and "month" not in frame.columns
    ]
    frames_with_year = [
        frame
        for frame in frames
        if "year" in frame.columns
        and "month" not in frame.columns
        and "quartal" not in frame.columns
    ]

    # Merge all dataframes on year and month
    joined = frames_with_month[0]
    for frame in frames_with_month[1:]:
        joined = pd.merge(
            joined, frame, on=["year", "month"], how="outer", suffixes=("", "_dup")
        )

    for frame in frames_with_quartal:
        joined = pd.merge(
            joined, frame, on=["year", "quartal"], how="outer", suffixes=("", "_dup")
        )

    for frame in frames_with_year:
        joined = pd.merge(
            joined, frame, on=["year"], how="outer", suffixes=("", "_dup")
        )
    # Drop duplicate columns if necessary
    joined = joined.loc[:, ~joined.columns.str.endswith("_dup")].drop(
        columns=["quartal"]
    )
    joined = joined.drop_duplicates(subset=["year", "month"])

    return joined
