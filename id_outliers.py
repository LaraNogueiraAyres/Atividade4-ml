import pandas as pd

df = pd.read_excel("RTVue_20221110_MLClass.xlsx")

cols = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]

outlier_report = {}

for col in cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df[f"{col}_outlier"] = (df[col] < lower) | (df[col] > upper)

    outlier_report[col] = {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower": lower,
        "upper": upper,
        "count": df[f"{col}_outlier"].sum()
    }

# Marca linhas com pelo menos um outlier
df["has_outlier"] = df[[f"{col}_outlier" for col in cols]].any(axis=1)

# Filtra apenas linhas suspeitas
outliers = df[df["has_outlier"]]

print(pd.DataFrame(outlier_report).T)
print(outliers)