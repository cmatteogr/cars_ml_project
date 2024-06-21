import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv(r'.\data\preprocess\cars_prepared.csv')

profile = ProfileReport(df, title="Cars Profiling Report")
profile.to_file("cars_report_prepared.html")
