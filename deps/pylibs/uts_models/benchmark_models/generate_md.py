import pandas as pd

from pylibs.utils.util_file import FileUtils

all = FileUtils().get_all_files("./", ext=".png")

pngs = []

for png in all:
    file = str(png).split("benchmark_models")[-1][1:]

    model_name = file.split("/")[0]
    pngs.append([model_name, file])
data = pd.DataFrame(pngs, columns=["name", "val"])

mds = []
for model, data_group in data.groupby("name"):
    mds.append(f"\n# {str(model).upper()}\n")
    for png in data_group.iterrows():
        mds.append(f"\nKPI ID = `{png[1].iloc[1].split('/')[-1][:-4]}`\n")
        mds.append(f"![{model}]({png[1].iloc[1]})\n")

with open("./views.md", "w") as f:
    f.writelines(mds)
