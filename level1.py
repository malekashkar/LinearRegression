import requests
import pandas as pd
import mysql.connector
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def get_data_1():
    indicators = {
        "GDP per capita (current US$)": "NY.GDP.PCAP.CD",
        "Labor force participation rate, male (% ages 15+)": "SL.TLF.CACT.MA.ZS",
        "Labor force participation rate, female (% ages 15+)": "SL.TLF.CACT.FE.ZS",
    }
    
    base_url = "https://api.worldbank.org/v2/country/all/indicator"
    data = []
    
    for indicator_name, indicator_code in indicators.items():
        url = f"{base_url}/{indicator_code}?format=json&per_page=5000"
        print(f"Fetching data for {indicator_name}...")
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Success for {indicator_name}")
            json_data = response.json()
            if len(json_data) > 1:
                for entry in json_data[1]:
                    if entry["value"] is not None:
                        data.append({
                            "Country": entry["country"]["value"],
                            "Year": entry["date"],
                            "Indicator": indicator_name,
                            "Value": entry["value"]
                        })
        else:
            print(f"Failed to retrieve data for indicator {indicator_name}. Status code: {response.status_code}")
    
    df = pd.DataFrame(data)
    print("Data fetched successfully.")
    print(f"Total records: {len(df)}")

    if not df.empty:
        df = df.pivot_table(index=["Country", "Year"], columns="Indicator", values="Value").reset_index()
        df.rename(columns={
            "GDP per capita (current US$)": "gdp_per_capita",
            "Labor force participation rate, male (% ages 15+)": "male_labor_force",
            "Labor force participation rate, female (% ages 15+)": "female_labor_force",
        }, inplace=True)

        df["difference_labor_force_participation_rate"] = (
            df["male_labor_force"] - df["female_labor_force"]
        )
    return df

def save_data_1(data, mydb, cursor, db_name):
    """
    Save the data in a MySQL database
    IN: data, DataFrame, data of GDP per capita, labor force participation rate of each country/territory
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
        CREATE TABLE IF NOT EXISTS gdp_per_capita (
            country_code VARCHAR(5),
            year INT,
            value FLOAT,
            PRIMARY KEY (country_code, year),
            FOREIGN KEY (country_code) REFERENCES country(code)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS labor_force_participation_rate (
            country_code VARCHAR(5),
            year INT,
            male_value FLOAT,
            female_value FLOAT,
            PRIMARY KEY (country_code, year),
            FOREIGN KEY (country_code) REFERENCES country(code)
        )
    """)

    countries = data[["Country"]].drop_duplicates().reset_index(drop=True)
    for _, row in countries.iterrows():
        country_name = row["Country"]
        country_code = country_name[:3].upper()
        cursor.execute("""
            INSERT IGNORE INTO country (code, name)
            VALUES (%s, %s)
        """, (country_code, country_name))

    gdp_data = data[["Country", "Year", "gdp_per_capita"]].dropna()
    for _, row in gdp_data.iterrows():
        country_code = row["Country"][:3].upper()
        year = int(row["Year"])
        value = float(row["gdp_per_capita"])
        cursor.execute("""
            INSERT IGNORE INTO gdp_per_capita (country_code, year, value)
            VALUES (%s, %s, %s)
        """, (country_code, year, value))

    labor_data = data[["Country", "Year", "male_labor_force", "female_labor_force"]].dropna()
    for _, row in labor_data.iterrows():
        country_code = row["Country"][:3].upper()
        year = int(row["Year"])
        male_value = float(row["male_labor_force"])
        female_value = float(row["female_labor_force"])
        cursor.execute("""
            INSERT IGNORE INTO labor_force_participation_rate (country_code, year, male_value, female_value)
            VALUES (%s, %s, %s, %s)
        """, (country_code, year, male_value, female_value))

    mydb.commit()
    print("Data saved successfully to the database.")

def load_data_1(mydb, cursor, db_name):
    """
    Query the data from the MySQL database
    IN: mydb, MySQL connection
        cursor, MySQL cursor
        db_name, str, name of the database
    OUT: DataFrame, data of GDP per capita, labor force participation rate of each country/territory
    """
    cursor.execute(f"USE {db_name}")

    query = """
        SELECT 
            c.name AS country_name,
            c.code AS country_code,
            g.year AS Year,
            g.value AS gdp_per_capita,
            (l.male_value - l.female_value) AS difference_labor_force_participation_rate
        FROM 
            country c
        INNER JOIN 
            gdp_per_capita g ON c.code = g.country_code
        INNER JOIN 
            labor_force_participation_rate l ON c.code = l.country_code AND g.year = l.year
        WHERE 
            g.value IS NOT NULL AND
            l.male_value IS NOT NULL AND
            l.female_value IS NOT NULL;
    """

    cursor.execute(query)
    result = cursor.fetchall()

    columns = [desc[0] for desc in cursor.description]

    df = pd.DataFrame(result, columns=columns)
    return df

def plot_data_1(data):
    data = data.dropna(subset=["gdp_per_capita", "difference_labor_force_participation_rate"])
    
    x = data["gdp_per_capita"]
    y = data["difference_labor_force_participation_rate"]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.title("Relationship Between GDP per Capita and Labor Force Participation Rate Difference")
    plt.xlabel("GDP per Capita (US$)")
    plt.ylabel("Difference in Labor Force Participation Rate (Male - Female)")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.show()

def compute_linear_regression_1(data):
    """
    Fit a linear regression line to the data
    IN: data, DataFrame, data of GDP per capita, labor force participation rate of each country/territory
    OUT: float, intercept of the regression line
         float, slope of the regression line
    """
    x = data["gdp_per_capita"]
    y = data["difference_labor_force_participation_rate"]
    
    x_mean = x.mean()
    y_mean = y.mean()
    
    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()
    slope = numerator / denominator
    
    intercept = y_mean - slope * x_mean
    
    return intercept, slope

def plot_linear_regression_1(data, slope, intercept):
    """
    Plot the data along with the linear regression line
    IN: data, DataFrame, data of GDP per capita, labor force participation rate of each country/territory
        slope, float, slope of the regression line
        intercept, float, intercept of the regression line
    OUT: None
    """
    x = data["gdp_per_capita"]
    y = data["difference_labor_force_participation_rate"]
    
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7, label="Data Points")
    plt.plot(x_line, y_line, color="red", label="Regression Line")
    plt.title("Linear Regression: GDP vs Labor Force Participation Rate Difference")
    plt.xlabel("GDP per Capita (US$)")
    plt.ylabel("Difference in Labor Force Participation Rate (Male - Female)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.show()

def get_determinant_1(data, slope, intercept):
    """
    Return the coefficient of determination of the linear regression
    IN: data, DataFrame, data of GDP per capita, labor force participation rate of each country/territory
        slope, float, slope of the regression line
        intercept, float, intercept of the regression line
    OUT: float, coefficient of determination
    """
    x = data["gdp_per_capita"]
    y = data["difference_labor_force_participation_rate"]
    
    y_pred = slope * x + intercept
    
    y_mean = y.mean()
    SST = ((y - y_mean) ** 2).sum()
    
    SSR = ((y_pred - y_mean) ** 2).sum()
    
    r_squared = SSR / SST
    
    return r_squared
