from level1 import get_data_1, save_data_1, load_data_1, compute_linear_regression_1, get_determinant_1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

def get_data_2():
    """
    Retrieve the GDP per capita, labor force participation rate of each country/territory.
    Ensure 'Year' is numeric.
    """
    data = get_data_1()
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")

    critical_columns = ["gdp_per_capita", "Year", "difference_labor_force_participation_rate"]
    data = data.dropna(subset=critical_columns)

    return data

def save_data_2(data, mydb, cursor, db_name):
    """
    Save the data in a MySQL database
    IN: data, DataFrame, data of GDP per capita, labor force participation rate of each country/territory
        mydb, MySQL connection
        cursor, MySQL cursor
        db_name, str, name of the database
    OUT: None
    """
    return save_data_1(data, mydb, cursor, db_name)

def load_data_2(mydb, cursor, db_name):
    """
    Load the data from the MySQL database and ensure 'Year' is numeric.
    """
    return load_data_1(mydb, cursor, db_name)

def plot_data_2(data):
    """
    Visualize the relationship between GDP per capita, the year, and the difference in labor force participation rates
    IN: data, DataFrame, data of GDP per capita, labor force participation rate of each country/territory
    OUT: None
    """
    X = data["gdp_per_capita"]
    Y = data["Year"]
    Z = data["difference_labor_force_participation_rate"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z, c='r', marker='o', alpha=0.7)

    ax.set_xlabel('GDP per capita (US$)')
    ax.set_ylabel('Year')
    ax.set_zlabel('Difference of Labor Force Participation Rate (Male - Female)')

    ax.set_title('3D Scatter Plot: GDP, Year, and Labor Force Participation Rate Difference')

    plt.show()

def compute_linear_regression_2(data):
    """
    Fit a linear regression plane to the data
    IN: data, DataFrame, data of GDP per capita, labor force participation rate of each country/territory
    OUT: ndarray of shape (3,), (intercept, slope_gdp, slope_year)
    """
    X1 = data["gdp_per_capita"]
    X2 = data["Year"]
    Y = data["difference_labor_force_participation_rate"]

    X = np.column_stack((np.ones(len(X1)), X1, X2))
    Y = Y.to_numpy()

    coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y

    return coefficients

def plot_linear_regression_2(data, regression_params):
    """
    Visualize the data as a 3D scatter plot along with the fitted linear regression plane
    IN: data, DataFrame, data of GDP per capita, labor force participation rate of each country/territory
        regression_params, ndarray of shape (3,), (intercept, slope_gdp, slope_year)
    OUT: None
    """
    intercept, slope_gdp, slope_year = regression_params

    X = data["gdp_per_capita"]
    Y = data["Year"]
    Z_actual = data["difference_labor_force_participation_rate"]

    X_range = np.linspace(X.min(), X.max(), 50)
    Y_range = np.linspace(Y.min(), Y.max(), 50)
    X_mesh, Y_mesh = np.meshgrid(X_range, Y_range)

    Z_mesh = slope_gdp * X_mesh + slope_year * Y_mesh + intercept

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z_actual, c='r', marker='o', label="Actual Data", alpha=0.7)

    ax.plot_surface(X_mesh, Y_mesh, Z_mesh, color='blue', alpha=0.5, label="Regression Plane")

    ax.set_xlabel('GDP per capita (US$)')
    ax.set_ylabel('Year')
    ax.set_zlabel('Difference in Labor Force Participation Rate')

    ax.set_title("3D Linear Regression: GDP, Year, and Labor Force Participation Rate Difference")

    ax.legend()

    plt.show()

def get_determinant_2(data, regression_params):
    """
    Return the coefficient of determination of the linear regression plane
    IN: data, DataFrame, data of GDP per capita, labor force participation rate of each country/territory
        regression_params, ndarray of shape (3,), (intercept, slope_gdp, slope_year)
    OUT: float, coefficient of determination
    """
    intercept, slope_gdp, slope_year = regression_params

    X = data["gdp_per_capita"]
    Y = data["Year"]
    Z_actual = data["difference_labor_force_participation_rate"]

    Z_pred = slope_gdp * X + slope_year * Y + intercept

    Z_mean = Z_actual.mean()
    SST = ((Z_actual - Z_mean) ** 2).sum()

    SSR = ((Z_pred - Z_mean) ** 2).sum()

    r_squared = SSR / SST

    return r_squared

def compare_determinants_2(data, regression_params):
    """
    Determine which factor—GDP per capita or year—is more influential in explaining the difference in labor force participation rates
    IN: data, DataFrame, data of GDP per capita, labor force participation rate of each country/territory
        regression_params, ndarray of shape (3,), (intercept, slope_gdp, slope_year)
    OUT: str, the more important factor
    """
    full_model_determinant = get_determinant_2(data, regression_params)
    
    gdp_data = data[["gdp_per_capita", "difference_labor_force_participation_rate"]]
    gdp_regression_params = compute_linear_regression_1(gdp_data)
    gdp_model_determinant = get_determinant_1(gdp_data, gdp_regression_params[1], gdp_regression_params[0])
    
    year_data = data[["Year", "difference_labor_force_participation_rate"]]
    year_regression_params = compute_linear_regression_1(year_data)
    year_model_determinant = get_determinant_1(year_data, year_regression_params[1], year_regression_params[0])
    
    gdp_difference = full_model_determinant - gdp_model_determinant
    year_difference = full_model_determinant - year_model_determinant
    
    if gdp_difference > year_difference:
        return "GDP per capita"
    else:
        return "Year"

