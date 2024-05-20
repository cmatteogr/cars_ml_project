
import pandas as pd
from ydata_profiling import ProfileReport


# df = pd.read_csv(r'C:\Users\cesar_0qb0xal\Documents\GitHub\cars_ml_project\data\data_exploration\input\cars.csv')
df = pd.read_csv(r'C:\Users\cesar_0qb0xal\Documents\GitHub\cars_ml_project\data\preprocess\cars_prepared.csv')

profile = ProfileReport(df, title="Cars Profiling Report")
profile.to_file("cars_report_prepared.html")