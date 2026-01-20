# %%

import pandas as pd

# %%

df = pd.read_csv("deforestation_dataset_PERU_imputed_coca.csv", sep=";")
df[df["NOMBDIST"] == "CHACHAPOYAS"].head()
# %%
pd.crosstab(df["Cluster"], [df["Región"]], values=df["Def_ha"], aggfunc="sum")

# %%
#
# # plot x axis years, def_ha on y and group by cluster only for Huanuco without the pivot
import matplotlib.pyplot as plt

# Filter for Huanuco
df_huanuco = df[df["NOMBDEP"].astype(str).str.upper() == "HUANUCO"].copy()

# Aggregate deforestation by YEAR and Cluster (no pivot)
g = (
    df_huanuco.groupby(["YEAR", "Cluster"], as_index=False)["Def_ha"]
    .sum()
    .sort_values(["Cluster", "YEAR"])
)

# Plot: x=YEAR, y=Def_ha, one line per Cluster
fig, ax = plt.subplots(figsize=(10, 6))
for cluster, d in g.groupby("Cluster"):
    ax.plot(
        d["YEAR"], d["Def_ha"], marker="o", linewidth=1.5, label=f"Cluster {cluster}"
    )

ax.set_title("Deforestation (ha) over Years by Cluster — Huanuco")
ax.set_xlabel("Year")
ax.set_ylabel("Deforested area (ha)")
ax.grid(True, alpha=0.3)
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %%
# pivot table
p = pd.pivot_table(
    df,
    index="Cluster",
    columns=["Región", "NOMBDEP", "YEAR"],
    values="Def_ha",
    aggfunc="sum",
)
print(p)
