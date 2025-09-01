#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 01 18:29:16 2025

@author: ndw
"""

import os
import pandas as pd
import re
import json

with open("dataset/votingDataset.json", "r") as f:
    data = f.read()

# Remove trailing commas before } or ]
cleaned = re.sub(r",\s*([}\]])", r"\1", data)

# Now parse
parsed = json.loads(cleaned)

# Save as CSV
import pandas as pd
df = pd.json_normalize(parsed)  # flatten if nested
df.to_csv("dataset/votingDataset.csv", index=False)


with open("dataset/votingDataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# If it's a dict of dicts → convert to dataframe
if isinstance(data, dict):
    df = pd.DataFrame.from_dict(data, orient="index")
else:
    df = pd.DataFrame(data)

