
import pandas as pd
from ydata_profiling import ProfileReport


df = pd.read_csv(r'C:\Users\cesar_0qb0xal\Documents\GitHub\cars_ml_project\data\data_exploration\input\cars.csv')

print(df.head(30))
df['exterior_color'] = df['exterior_color'].map(lambda ec: ec.replace(' ', '_').lower() if not pd.isna(ec) else ec)
df['interior_color'] = df['interior_color'].map(lambda ec: ec.replace(' ', '_').lower() if not pd.isna(ec) else ec)

profile = ProfileReport(df, title="Cars Profiling Report")
profile.to_file("cars_report.html")