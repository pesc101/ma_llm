# %%
import pandas as pd

# Read the Excel file
root_folder = ""
df = pd.read_excel(f"{root_folder}/ma_llm/evaluation/results/spyder.xlsx")
df_created = pd.read_json(
    f"{root_folder}/ma_llm/evaluation/results/spyder_question_answer_pairs.jsonl",
    lines=True,
)

# %%

## Distribution of the category column
print(df["category"].value_counts())

print()
## Distribution of the file_name column
print(f'Number of Files used: {len(df["file_name"].unique())}')
print()

## df where category is file_import_from and then the distribution of the artifact_type column
df_file_import_from = df[df["category"] == "file_import_from"]
print(df_file_import_from["artifact_type"].value_counts())
