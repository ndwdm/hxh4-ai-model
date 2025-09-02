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

# If it's a dict of dicts → convert to dataframe
if isinstance(data, dict):
    df = pd.DataFrame.from_dict(parsed, orient="index")
else:
    df = pd.DataFrame(parsed)


# Save as CSV
df.to_csv("dataset/votingDataset.csv", index=False)
print("✅ JSON cleaned and converted to CSV successfully.")

