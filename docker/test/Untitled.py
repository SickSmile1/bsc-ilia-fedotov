# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np

# %%
a = np.linspace(1,100,100)
a = a.reshape((1,10,10))

# %%
np.shape(a)

# %%
print(np.isclose(a[0], 0))

# %%
