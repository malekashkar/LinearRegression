import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data_3(x_indicator_codes):
    """
    Retrieve the data of each country/territory
    IN: x_indicator_codes, list of str, indicator codes as predictors for the linear regression
    OUT: DataFrame, data of each country/territory
    """
    base_url = "https://api.worldbank.org/v2/country/all/indicator"
    data = []

    for indicator in x_indicator_codes:
        url = f"{base_url}/{indicator}?format=json&per_page=5000"
        print(f"Fetching data for indicator: {indicator}")
        response = requests.get(url)
        if response.status_code == 200:
            json_data = response.json()
            if len(json_data) > 1:
                for entry in json_data[1]:
                    if entry["value"] is not None:
                        data.append({
                            "Country": entry["country"]["value"],
                            "Year": entry["date"],
                            "Indicator": indicator,
                            "Value": entry["value"]
                        })
        else:
            print(f"Failed to retrieve data for indicator {indicator}. Status code: {response.status_code}")
    
    df = pd.DataFrame(data)
    df = df.pivot_table(index=["Country", "Year"], columns="Indicator", values="Value").reset_index()

    df = df.rename(columns={
        "NY.GDP.PCAP.CD": "gdp_per_capita",
        "SL.TLF.CACT.MA.ZS": "labor_force_male",
        "SL.TLF.CACT.FE.ZS": "labor_force_female",
        "SL.TLF.TOTL.FE.ZS": "labor_force_female_total",
        "SE.ADT.LITR.MA.ZS": "literacy_rate_male",
        "SE.ADT.LITR.FE.ZS": "literacy_rate_female",
        "SP.DYN.LE00.MA.IN": "life_expectancy_male",
        "SP.DYN.LE00.FE.IN": "life_expectancy_female"
    })

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    df = df.fillna(0)

    return df

def save_data_3(data, mydb, cursor, db_name):
    """
    Save the data in a MySQL database
    IN: data, DataFrame, data of each country/territory
        mydb, MySQL connection
        cursor, MySQL cursor
        db_name, str, name of the database
    OUT: None
    """
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    cursor.execute(f"USE {db_name}")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS country (
            code VARCHAR(5) PRIMARY KEY,
            name VARCHAR(255) NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS indicators (
            country_code VARCHAR(5),
            year INT,
            gdp_per_capita FLOAT,
            labor_force_male FLOAT,
            labor_force_female FLOAT,
            labor_force_female_total FLOAT,
            literacy_rate_male FLOAT,
            literacy_rate_female FLOAT,
            life_expectancy_male FLOAT,
            life_expectancy_female FLOAT,
            PRIMARY KEY (country_code, year),
            FOREIGN KEY (country_code) REFERENCES country(code)
        )
    """)

    indicator_columns = [
        "NY.GDP.PCAP.CD", "SL.TLF.CACT.MA.ZS", "SL.TLF.CACT.FE.ZS",
        "SL.TLF.TOTL.FE.ZS", "SE.ADT.LITR.MA.ZS", "SE.ADT.LITR.FE.ZS",
        "SP.DYN.LE00.MA.IN", "SP.DYN.LE00.FE.IN"
    ]
    data = data.fillna(0)

    countries = data[["Country"]].drop_duplicates().reset_index(drop=True)
    for _, row in countries.iterrows():
        country_name = row["Country"]
        country_code = country_name[:3].upper()
        cursor.execute("""
            INSERT IGNORE INTO country (code, name)
            VALUES (%s, %s)
        """, (country_code, country_name))

    for _, row in data.iterrows():
        country_code = row["Country"][:3].upper()
        year = int(row["Year"])
        cursor.execute("""
            INSERT IGNORE INTO indicators (
                country_code, year, gdp_per_capita, labor_force_male, labor_force_female, 
                labor_force_female_total, literacy_rate_male, literacy_rate_female,
                life_expectancy_male, life_expectancy_female
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            country_code,
            year,
            row.get("NY.GDP.PCAP.CD"),
            row.get("SL.TLF.CACT.MA.ZS"),
            row.get("SL.TLF.CACT.FE.ZS"),
            row.get("SL.TLF.TOTL.FE.ZS"),
            row.get("SE.ADT.LITR.MA.ZS"),
            row.get("SE.ADT.LITR.FE.ZS"),
            row.get("SP.DYN.LE00.MA.IN"),
            row.get("SP.DYN.LE00.FE.IN") 
        ))

    mydb.commit()
    print("Data saved successfully!")

def load_data_3(mydb, cursor, db_name):
    """
    Query the data from the MySQL database
    IN: mydb, MySQL connection
        cursor, MySQL cursor
        db_name, str, name of the database
    OUT: DataFrame, data of each country/territory
    """
    cursor.execute(f"USE {db_name}")

    query = """
        SELECT 
            c.name AS country_name,
            c.code AS country_code,
            i.year,
            i.gdp_per_capita,
            i.labor_force_male,
            i.labor_force_female,
            i.labor_force_female_total,
            i.literacy_rate_male,
            i.literacy_rate_female,
            i.life_expectancy_male,
            i.life_expectancy_female
        FROM 
            country c
        INNER JOIN 
            indicators i ON c.code = i.country_code
        WHERE 
            i.gdp_per_capita IS NOT NULL AND
            i.labor_force_male IS NOT NULL AND
            i.labor_force_female IS NOT NULL
    """

    cursor.execute(query)

    result = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(result, columns=columns)

    return df

def plot_data_3(data):
    """
    Visualize the pairwise relationships between each factor and the difference in labor force participation rate
    IN: data, DataFrame, data of each country/territory
    OUT: None
    """
    data = data.rename(columns={
        "NY.GDP.PCAP.CD": "gdp_per_capita",
        "SL.TLF.CACT.MA.ZS": "labor_force_male",
        "SL.TLF.CACT.FE.ZS": "labor_force_female",
        "SL.TLF.TOTL.FE.ZS": "labor_force_female_total",
        "SE.ADT.LITR.MA.ZS": "literacy_rate_male",
        "SE.ADT.LITR.FE.ZS": "literacy_rate_female",
        "SP.DYN.LE00.MA.IN": "life_expectancy_male",
        "SP.DYN.LE00.FE.IN": "life_expectancy_female"
    })
    
    factors = [
        "gdp_per_capita",
        "labor_force_male",
        "labor_force_female",
        "labor_force_female_total",
        "literacy_rate_male",
        "literacy_rate_female",
        "life_expectancy_male",
        "life_expectancy_female"
    ]

    num_factors = len(factors)
    fig, axes = plt.subplots(num_factors, 1, figsize=(8, 4 * num_factors), sharex=False)

    for i, factor in enumerate(factors):
        ax = axes[i]
        ax.scatter(data[factor], data["labor_force_female"] - data["labor_force_male"], alpha=0.7)
        ax.set_title(f"{factor} vs. Difference in Labor Force Participation Rate")
        ax.set_xlabel(factor)
        ax.set_ylabel("Difference in Labor Force Rate")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

def compute_linear_regression_3(data):
    """
    Fit a high-dimensional hyperplane to model the relationship between all factors and the difference in labor force participation rate
    IN: data, DataFrame, data of each country/territory
    OUT: ndarray of shape (m+1,), (intercept, slope_1, slope_2, ..., slope_n)
    """
    data = data.rename(columns={
        "NY.GDP.PCAP.CD": "gdp_per_capita",
        "SL.TLF.CACT.MA.ZS": "labor_force_male",
        "SL.TLF.CACT.FE.ZS": "labor_force_female",
        "SL.TLF.TOTL.FE.ZS": "labor_force_female_total",
        "SE.ADT.LITR.MA.ZS": "literacy_rate_male",
        "SE.ADT.LITR.FE.ZS": "literacy_rate_female",
        "SP.DYN.LE00.MA.IN": "life_expectancy_male",
        "SP.DYN.LE00.FE.IN": "life_expectancy_female"
    })

    factors = [
        "gdp_per_capita",
        "labor_force_male",
        "labor_force_female",
        "labor_force_female_total",
        "literacy_rate_male",
        "literacy_rate_female",
        "life_expectancy_male",
        "life_expectancy_female"
    ]

    X = data[factors].to_numpy()
    X = np.column_stack((np.ones(X.shape[0]), X))

    y = (data["labor_force_female"] - data["labor_force_male"]).to_numpy()

    coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    return coefficients

def get_determinant_3(data, regression_params):
    """
    Return the coefficient of determination of the linear regression hyperplane
    IN: data, DataFrame, data of each country/territory
        regression_params, ndarray of shape (m+1,), (intercept, slope_1, slope_2, ..., slope_n)
    OUT: float, coefficient of determination
    """
    data = data.rename(columns={
        "NY.GDP.PCAP.CD": "gdp_per_capita",
        "SL.TLF.CACT.MA.ZS": "labor_force_male",
        "SL.TLF.CACT.FE.ZS": "labor_force_female",
        "SL.TLF.TOTL.FE.ZS": "labor_force_female_total",
        "SE.ADT.LITR.MA.ZS": "literacy_rate_male",
        "SE.ADT.LITR.FE.ZS": "literacy_rate_female",
        "SP.DYN.LE00.MA.IN": "life_expectancy_male",
        "SP.DYN.LE00.FE.IN": "life_expectancy_female"
    })

    factors = [
        "gdp_per_capita",
        "labor_force_male",
        "labor_force_female",
        "labor_force_female_total",
        "literacy_rate_male",
        "literacy_rate_female",
        "life_expectancy_male",
        "life_expectancy_female"
    ]

    intercept = regression_params[0]
    slopes = regression_params[1:]

    X = data[factors].to_numpy()

    y_actual = (data["labor_force_female"] - data["labor_force_male"]).to_numpy()

    y_pred = intercept + X @ slopes

    y_mean = y_actual.mean()
    SST = ((y_actual - y_mean) ** 2).sum()

    SSR = ((y_pred - y_mean) ** 2).sum()

    r_squared = SSR / SST

    return r_squared

def compare_determinants_3(data, regression_params):
    """
    Determine the factors in non-increasing order of importance in explaining the difference in labor force participation rate
    IN: data, DataFrame, data of each country/territory
        regression_params, ndarray of shape (m+1,), (intercept, slope_1, slope_2, ..., slope_n)
    OUT: list of str, the names of the factors in non-increasing order of importance
    """
    data = data.rename(columns={
        "NY.GDP.PCAP.CD": "gdp_per_capita",
        "SL.TLF.CACT.MA.ZS": "labor_force_male",
        "SL.TLF.CACT.FE.ZS": "labor_force_female",
        "SL.TLF.TOTL.FE.ZS": "labor_force_female_total",
        "SE.ADT.LITR.MA.ZS": "literacy_rate_male",
        "SE.ADT.LITR.FE.ZS": "literacy_rate_female",
        "SP.DYN.LE00.MA.IN": "life_expectancy_male",
        "SP.DYN.LE00.FE.IN": "life_expectancy_female"
    })

    factors = [
        "gdp_per_capita",
        "labor_force_male",
        "labor_force_female",
        "labor_force_female_total",
        "literacy_rate_male",
        "literacy_rate_female",
        "life_expectancy_male",
        "life_expectancy_female"
    ]

    full_model_r_squared = get_determinant_3(data, regression_params)

    importance_scores = {}

    for factor in factors:
        reduced_factors = [f for f in factors if f != factor]
        X_reduced = data[reduced_factors].to_numpy()
        X_reduced = np.column_stack((np.ones(X_reduced.shape[0]), X_reduced))
        y = (data["labor_force_female"] - data["labor_force_male"]).to_numpy()

        reduced_regression_params = np.linalg.inv(X_reduced.T @ X_reduced) @ X_reduced.T @ y

        reduced_r_squared = get_determinant_3(data, reduced_regression_params)

        importance_scores[factor] = full_model_r_squared - reduced_r_squared

    sorted_factors = sorted(importance_scores, key=importance_scores.get, reverse=True)

    return sorted_factors

def plot_linear_regression_3(data, regression_params, important_factors):
    """
    Visualize the data as a 3D scatter plot along with the fitted linear regression plane
    IN: data, DataFrame, data of each country/territory
        regression_params, ndarray of shape (m+1,), (intercept, slope_1, slope_2, ..., slope_n)
        important_factors, list of str, the names of the two most important factors
    OUT: None
    """
    data = data.rename(columns={
        "NY.GDP.PCAP.CD": "gdp_per_capita",
        "SL.TLF.CACT.MA.ZS": "labor_force_male",
        "SL.TLF.CACT.FE.ZS": "labor_force_female",
        "SL.TLF.TOTL.FE.ZS": "labor_force_female_total",
        "SE.ADT.LITR.MA.ZS": "literacy_rate_male",
        "SE.ADT.LITR.FE.ZS": "literacy_rate_female",
        "SP.DYN.LE00.MA.IN": "life_expectancy_male",
        "SP.DYN.LE00.FE.IN": "life_expectancy_female"
    })

    factor_x, factor_y = important_factors

    X = data[factor_x]
    Y = data[factor_y]
    Z_actual = data["labor_force_female"] - data["labor_force_male"]

    intercept = regression_params[0]
    slopes = regression_params[1:]
    slope_x = slopes[data.columns.get_loc(factor_x) - 1]
    slope_y = slopes[data.columns.get_loc(factor_y) - 1]

    X_range = np.linspace(X.min(), X.max(), 50)
    Y_range = np.linspace(Y.min(), Y.max(), 50)
    X_mesh, Y_mesh = np.meshgrid(X_range, Y_range)

    Z_mesh = intercept + slope_x * X_mesh + slope_y * Y_mesh

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z_actual, c='r', marker='o', label="Actual Data", alpha=0.7)

    ax.plot_surface(X_mesh, Y_mesh, Z_mesh, color='blue', alpha=0.5, label="Regression Plane")

    ax.set_xlabel(factor_x)
    ax.set_ylabel(factor_y)
    ax.set_zlabel('Difference in Labor Force Participation Rate')

    ax.set_title(f"3D Linear Regression: {factor_x}, {factor_y}, and Labor Force Participation Rate Difference")

    ax.legend()

    plt.show()