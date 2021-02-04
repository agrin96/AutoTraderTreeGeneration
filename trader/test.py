import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("../indicator_data.csv")
    days = 7
    slice_index = 86400*days
    df = df.iloc[:slice_index]
    df.to_csv("../first_week_indicator_data.csv")
