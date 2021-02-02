#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import matplotlib.pyplot as plt


# In[5]:


df = pandas.DataFrame.from_csv("NALA_CALA_W_25.csv")
df.sort_index()
print(len(df))


# In[22]:


fig, ax = plt.subplots()

text_size = 16

ax.scatter(df["phi"], df["psi"], s=2)

ax.plot([-180, 180], [0, 0], "k:")
ax.plot([0, 0], [-180, 180], "k:")

plt.xlabel("$\Phi$", fontsize=text_size)
plt.ylabel("$\Psi$", fontsize=text_size)

plt.xlim(-180, 180)
plt.ylim(-180, 180)
plt.gca().set_aspect('equal', adjustable='box')

plt.savefig("output.png", dpi=1000, bbox="tight")

