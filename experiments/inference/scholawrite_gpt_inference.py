# -*- coding: utf-8 -*-
"""ScholaWrite-GPT-Inference

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/astrohl/scholawrite-gpt-inference.dccf5d4b-8670-42a3-bbde-42369d740b19.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20250610/auto/storage/goog4_request%26X-Goog-Date%3D20250610T182728Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D8d4ca8b78ef94c8e8ee64e86f537bbae9d74af62d22b90837c6f4df0695de5752167b4fd3d1bfd57cdf6fce42313d15402a9d2c6cd84874fefd4882bf58595c31c56441e67efcf99a28d8903edc86b3571691455e2d874cef89168a72d03268e2e5081b54758a9d6bb8e1c2942a5d4245fb1f19aa8cbe8de767080f081693cdb2492afa073c53817870a55335f490583e5f867ab3ef869ce1c22f1016f0a69848692a4caa5ce9ff2a5c5ba61885edc8ccb75027baca9aac4502b9ec907de68aac4758ab2ba10fe8699ad76e9a8f289db3567f2a48409d09fd969e6f006bc346d9e332c66f87cff1a8d9fe37be75d73a8ae7ab3dba2bf977fb73c3eab522c5d7f
"""

# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

astrohl_scholawrite_path = kagglehub.dataset_download('astrohl/scholawrite')

print('Data source import complete.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pls

data1 = pd.read_parquet('/kaggle/input/scholawrite/all_sorted-00000-of-00002.parquet', engine='pyarrow')
data2 = pd.read_parquet('/kaggle/input/scholawrite/all_sorted-00001-of-00002.parquet', engine='pyarrow')
test_data1 = pd.read_parquet('/kaggle/input/scholawrite/test-00000-of-00001.parquet', engine='pyarrow')
test_data2 = pd.read_parquet('/kaggle/input/scholawrite/test_small-00000-of-00001.parquet', engine='pyarrow')
train_data2 = pd.read_parquet('/kaggle/input/scholawrite/train-00000-of-00001.parquet', engine='pyarrow')

test_data1.shape, test_data2.shape, train_data2.shape

train_data2.head()

examples = ""

for i in range(3):
    examples += ("Input: \n" + train_data2['before text'][i] + " \nLabel: \n" + train_data2['label'][i] + "\n")
    # print("Input: \n", train_data2['before text'][i], " \nLabel: \n", train_data2['label'][i], "\n")
examples

def class_prompt(before_text):
    system_prompt= """You are a classifier that identify the most likely next writing intention. You will be given a list of all possible writing intention labels with definitions, and an in-progress LaTex paper draft written by a graduate student. Please strictly follow user's instruction to identify the most likely next writing intention"""

    usr_prompt= f"""Here is a list of all the possible writing intention labels with definitions:

Idea Generation: Formulate and record initial thoughts and concepts.
Idea Organization: Select the most useful materials and demarcate those generated ideas in a visually formatted way.
Section Planning: Initially create sections and sub-level structures.
Text Production: Translate their ideas into full languages, either from the writers' language or borrowed sentences from an external source.
Object Insertion: Insert visual claims of their arguments (e.g., figures, tables, equations, footnotes, itemized lists, etc.).
Cross-reference: Link different sections, figures, tables, or other elements within a document through referencing commands.
Citation Integration: Incorporate bibliographic references into a document and systematically link these references using citation commands.
Macro Insertion: Incorporate predefined commands or packages into a LaTeX document to alter its formatting.
Fluency: Fix grammatical or syntactic errors in the text or LaTeX commands.
Coherence: Logically link (1) any of the two or multiple sentences within the same paragraph; (2) any two subsequent paragraphs; or (3) objects to be consistent as a whole.
Structural: Improve the flow of information by modifying the location of texts and objects.
Clarity: Improve the semantic relationships between texts to be more straightforward and concise.
Linguistic Style: Modify texts with the writer's writing preferences regarding styles and word choices, etc.
Scientific Accuracy: Update or correct scientific evidence (e.g., numbers, equations) for more accurate claims.
Visual Formatting: Modify the stylistic formatting of texts, objects, and citations.

Identify the most likely next writing intention of a graduate researcher when writing the following LaTex paper draft. Your output should only be a label from the list above.

Here is LaTeX paper draft:
{before_text}

Here're some examples for your references

{examples}

"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": usr_prompt}
    ]

!pip install dotenv

import openai
print(openai.__version__)

from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key="OPENAI_API")

def process_label(predicted_label):
    all_labels = ['Text Production', 'Visual Formatting', 'Clarity', 'Section Planning',
                 'Structural', 'Object Insertion', 'Cross-reference', 'Fluency',
                 'Idea Generation', 'Idea Organization', 'Citation Integration', 'Coherence',
                 'Linguistic Style', 'Scientific Accuracy', 'Macro Insertion']
    if predicted_label not in all_labels:
        for true_label in all_labels:
            if true_label in predicted_label:
                return true_label
        print("Unexpected label:", predicted_label)
    return predicted_label

def get_label(before_text):
    prompt = class_prompt(before_text)  # ensure this returns a list of {"role": ..., "content": ...}
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=prompt
    )
    generated_response = response.choices[0].message.content
    return process_label(generated_response)

def main():
    results = []
    for before_text in tqdm(test_data2["before text"].values):
        writing_intention = get_label(before_text)
        print('Result:', writing_intention)
        results.append(writing_intention)

    test_data2["predicted"] = results
    test_data2.to_csv("/kaggle/working/gpt4o_class_result.csv", index=False)

accuracy = (test_data2["predicted"] == test_data2["label"]).mean()
print(f"Accuracy: {accuracy:.4f}")

from sklearn.metrics import f1_score

f1 = f1_score(test_data2["label"], test_data2["predicted"], average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")

main()