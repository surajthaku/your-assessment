import pandas as pd
df = pd.read_csv("test_data.csv")
df.sample(100).to_json("baseline_data.json", orient="records")
