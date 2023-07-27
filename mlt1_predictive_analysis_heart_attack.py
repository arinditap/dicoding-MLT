# -*- coding: utf-8 -*-
"""MLT1-Predictive-Analysis-Heart-Attack.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ErRFdbXmjtCdLUROQbaWAYpDpCv_I7vV

Prediksi Potensi Penyakit Jantung

1. Persiapan [Preparation]

> 1.1. Import modul dan package yang akan digunakan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

"""> 1.2. Mempersiapkan dataset yang akan digunakan dengan menginstall kaggle"""

!pip install -q kaggle
from google.colab import files
files.upload() #upload kaggle.json

!rm -r ~/.kaggle
!mkdir ~/.kaggle
!mv ./kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

#mengudnuh dataset dari kaggle
!kaggle datasets download -d johnsmith88/heart-disease-dataset

#ekstrak file zip ke direktori aktif
!unzip /content/heart-disease-dataset.zip

"""2. Data Understanding
> 2.1. Memuat dataset ke dalam dataframe menggunakan pandas
"""

df = pd.read_csv("/content/heart.csv")

df.shape

"""Didapatkan bahwa terdapat 1025 baris dan 14 kolom"""

df.head()

#memuat informasi dataframe
df.info()

#memuat deskripsi tiap kolom ygg kemudian ditranspose untuk memudahkan pembacaan
df.describe().T

"""> 2.2. Cek missing values atau data yg kosong

"""

#melihat apakah ada data yg kosong di tiap kolomnya
df.isnull().sum()

"""Dalam data tidak terdapat data yang kosong atau missing values

> 2.3. Memisahkan categorical dan continuous column
"""

cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
cont_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
target_col = ['target']

print('Yang termasuk kolom kategorial adalah : ', cat_cols)
print('Yang termasuk kolom kontinyu adalah : ', cont_cols)
print('Yang termasuk kolom target adalah : ', target_col)

df[cont_cols].describe().T

"""3. Exploratory Data Analysis (EDA)
> 3.1. Univariate Analysis

>>> 3.1.1. Visualisasi data berupa count plot untuk fitur kategorikal (categorical features)
"""

fig = plt.figure(figsize=(10,7))
g = fig.add_gridspec(3,3)
g.update(wspace=0.5, hspace=0.25)
ax0 = fig.add_subplot(g[0,0])
ax1 = fig.add_subplot(g[0,1])
ax2 = fig.add_subplot(g[0,2])
ax3 = fig.add_subplot(g[1,0])
ax4 = fig.add_subplot(g[1,1])
ax5 = fig.add_subplot(g[1,2])
ax6 = fig.add_subplot(g[2,0])
ax7 = fig.add_subplot(g[2,1])
ax8 = fig.add_subplot(g[2,2])

color_palette = ['#800000', '#1d3557','#8000ff', '#6aac90', '#5833ff', '#da8829']

ax0.spines['bottom'].set_visible(False)
ax0.spines['left'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.tick_params(left=False, bottom=False)
ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.text(0.5, 0.5, "Count plot untuk\n jumlah fitur kategorial\n",
         horizontalalignment='center',
         verticalalignment='center',
         fontsize='14', fontweight='bold',
         fontfamily='serif', color='#000000')

# Sex count
ax1.text(0.3, 730, 'Sex', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax1,data=df,x='sex',palette=color_palette)
ax1.set_xlabel("")
ax1.set_ylabel("")

# Cp count
ax2.text(1.5, 525, 'Cp', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax2.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax2,data=df,x='cp',palette=color_palette)
ax2.set_xlabel("")
ax2.set_ylabel("")

# Fbs count
ax3.text(0.3, 900, 'Fbs', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax3,data=df,x='fbs',palette=color_palette)
ax3.set_xlabel("")
ax3.set_ylabel("")

# Restecg count
ax4.text(0.3, 530, 'Restecg', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax4.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax4,data=df,x='restecg',palette=color_palette)
ax4.set_xlabel("")
ax4.set_ylabel("")

# Exang count
ax5.text(0.3, 725, 'Exang', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax5.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax5,data=df,x='exang',palette=color_palette)
ax5.set_xlabel("")
ax5.set_ylabel("")

# slope count
ax6.text(0.5, 525, 'Slope', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax6.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax6,data=df,x='slope',palette=color_palette)
ax6.set_xlabel("")
ax6.set_ylabel("")

# ca count
ax7.text(1.5, 625, 'Ca', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax7.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax7,data=df,x='ca',palette=color_palette)
ax7.set_xlabel("")
ax7.set_ylabel("")

# thal count
ax8.text(0.85, 575, 'Thal', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax8.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax8,data=df,x='thal',palette=color_palette)
ax8.set_xlabel("")
ax8.set_ylabel("")

for s in ["top","right","left"]:
    ax1.spines[s].set_visible(False)
    ax2.spines[s].set_visible(False)
    ax3.spines[s].set_visible(False)
    ax4.spines[s].set_visible(False)
    ax5.spines[s].set_visible(False)
    ax6.spines[s].set_visible(False)
    ax7.spines[s].set_visible(False)
    ax8.spines[s].set_visible(False)

""">>> 3.1.2. Visualisasi data berupa boxenplot untuk fitur kontinyu (continuous features)"""

fig = plt.figure(figsize=(12,8))
g = fig.add_gridspec(2,3)
g.update(wspace=0.5, hspace=0.25)
ax0 = fig.add_subplot(g[0,0])
ax1 = fig.add_subplot(g[0,1])
ax2 = fig.add_subplot(g[0,2])
ax3 = fig.add_subplot(g[1,0])
ax4 = fig.add_subplot(g[1,1])
ax5 = fig.add_subplot(g[1,2])

color_palette = ['#800000', '#1d3557','#8000ff', '#6aac90', '#5833ff', '#da8829']

#title
ax0.spines['bottom'].set_visible(False)
ax0.spines['left'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.tick_params(left=False, bottom=False)
ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.text(0.5, 0.5, "Boxen Plot\nuntuk fitur\nkontinyu\n",
        horizontalalignment='center',
        verticalalignment='center',
        fontsize='14', fontweight='bold',
        fontfamily='serif', color='#000000')

# Age count
ax1.text(-0.05, 81, 'Age', fontsize=10, fontweight='bold', fontfamily='serif', color="#000000")
ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.boxenplot(ax=ax1,data=df,y=df['age'],palette=["#800000"], width=0.6)
ax1.set_xlabel("")
ax1.set_ylabel("")

# Trestbps count
ax2.text(-0.05, 210, 'Trestbps', fontsize=10, fontweight='bold', fontfamily='serif', color="#000000")
ax2.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.boxenplot(ax=ax2,data=df,y=df['trestbps'],palette=['#1d3557'], width=0.6)
ax2.set_xlabel("")
ax2.set_ylabel("")

# Chol count
ax3.text(-0.05, 600, 'Chol', fontsize=10, fontweight='bold', fontfamily='serif', color="#000000")
ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.boxenplot(ax=ax3,data=df,y=df['chol'],palette=["#8000ff"], width=0.6)
ax3.set_xlabel("")
ax3.set_ylabel("")

# Thalach count
ax4.text(-0.05, 210, 'Thalach', fontsize=10, fontweight='bold', fontfamily='serif', color="#000000")
ax4.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.boxenplot(ax=ax4,data=df,y=df['thalach'],palette=["#6aac90"], width=0.6)
ax4.set_xlabel("")
ax4.set_ylabel("")

# Oldpeak count
ax5.text(-0.09, 6.5, 'Oldpeak', fontsize=10, fontweight='bold', fontfamily='serif', color="#000000")
ax5.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.boxenplot(ax=ax5,data=df,y=df['oldpeak'],palette=["#5833ff"], width=0.6)
ax5.set_xlabel("")
ax5.set_ylabel("")

for s in ["top","right","left"]:
    ax1.spines[s].set_visible(False)
    ax2.spines[s].set_visible(False)
    ax3.spines[s].set_visible(False)
    ax4.spines[s].set_visible(False)
    ax5.spines[s].set_visible(False)

""">>> 3.1.3. Visualisasi data berupa countplot untuk kolom target"""

fig = plt.figure(figsize=(12,4))
g = fig.add_gridspec(1,2)
g.update(wspace=0.5, hspace=0.15)
ax0 = fig.add_subplot(g[0,0])
ax1 = fig.add_subplot(g[0,1])

color_palette = ["#800000","#8000ff","#6aac90","#5833ff","#da8829"]

# Title of the plot
ax0.text(0.5,0.5,"Count plot untuk target\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 14,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')

ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.tick_params(left=False, bottom=False)

# Target Count
ax1.text(0.35,550,"Target",fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax1, data=df, x = 'target',palette = color_palette)
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.set_xticklabels(["Low chances of attack(0)","High chances of attack(1)"])

ax0.spines["top"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax0.spines["bottom"].set_visible(False)
ax0.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)

""">3.2. Multivariate Analysis
>>>3.2.1. Correlation matrix dan heatmap
"""

df_corr = df[cont_cols].corr().T
df_corr

df_corr = df[cont_cols].corr().T
plt.figure(figsize=(10,8))
correlation_matrix = df.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

"""- Nilai korelasi semakin mendekati 1 maka semakin tinggi tingkat korelasinya
- Nilai korelasi semakin mendekati -1 maka semakin rendah tingkat korelasinya
- Faktor yang memiliki korelasi tinggi adalah 'cp', 'thalach', dan 'slope'
- Faktor yang memiliki korelasi rendah adalah 'restecg', dan 'fbs'

>>>3.2.2. Distribusi fitur kontinyu terhadap variabel target
"""

fig = plt.figure(figsize=(10,10))
g = fig.add_gridspec(5,2)
g.update(wspace=0.5, hspace=0.5)
ax0 = fig.add_subplot(g[0,0])
ax1 = fig.add_subplot(g[0,1])
ax2 = fig.add_subplot(g[1,0])
ax3 = fig.add_subplot(g[1,1])
ax4 = fig.add_subplot(g[2,0])
ax5 = fig.add_subplot(g[2,1])
ax6 = fig.add_subplot(g[3,0])
ax7 = fig.add_subplot(g[3,1])
ax8 = fig.add_subplot(g[4,0])
ax9 = fig.add_subplot(g[4,1])

# Age title
ax0.text(0.5,0.5,"Distribusi 'age'\nterhadap\n target variable\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 14,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax0.spines["bottom"].set_visible(False)
ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.tick_params(left=False, bottom=False)

# Age
ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax1, data=df, x='age',hue="target", fill=True,palette=["#8000ff","#da8829"], alpha=.5, linewidth=0)
ax1.set_xlabel("")
ax1.set_ylabel("")

# Trestbps title
ax2.text(0.5,0.5,"Distribusi 'trestbps'\nterhadap\n target variable\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 14,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax2.spines["bottom"].set_visible(False)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.tick_params(left=False, bottom=False)

# TrTbps
ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax3, data=df, x='trestbps',hue="target", fill=True,palette=["#8000ff","#da8829"], alpha=.5, linewidth=0)
ax3.set_xlabel("")
ax3.set_ylabel("")

# Chol title
ax4.text(0.5,0.5,"Distribusi 'chol'\nterhadap\n target variable\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 14,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax4.spines["bottom"].set_visible(False)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.tick_params(left=False, bottom=False)

# Chol
ax5.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax5, data=df, x='chol',hue="target", fill=True,palette=["#8000ff","#da8829"], alpha=.5, linewidth=0)
ax5.set_xlabel("")
ax5.set_ylabel("")

# Thalach title
ax6.text(0.5,0.5,"Distribusi 'thalach'\nterhadap\n target variable\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 14,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax6.spines["bottom"].set_visible(False)
ax6.set_xticklabels([])
ax6.set_yticklabels([])
ax6.tick_params(left=False, bottom=False)

# Thalach
ax7.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax7, data=df, x='thalach',hue="target", fill=True,palette=["#8000ff","#da8829"], alpha=.5, linewidth=0)
ax7.set_xlabel("")
ax7.set_ylabel("")

# Oldpeak title
ax8.text(0.5,0.5,"Distribuso 'oldpeak'\nterhadap\n target variable\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 14,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax8.spines["bottom"].set_visible(False)
ax8.set_xticklabels([])
ax8.set_yticklabels([])
ax8.tick_params(left=False, bottom=False)

# Oldpeak
ax9.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax9, data=df, x='oldpeak',hue="target", fill=True,palette=["#8000ff","#da8829"], alpha=.5, linewidth=0)
ax9.set_xlabel("")
ax9.set_ylabel("")

for i in ["top","left","right"]:
    ax0.spines[i].set_visible(False)
    ax1.spines[i].set_visible(False)
    ax2.spines[i].set_visible(False)
    ax3.spines[i].set_visible(False)
    ax4.spines[i].set_visible(False)
    ax5.spines[i].set_visible(False)
    ax6.spines[i].set_visible(False)
    ax7.spines[i].set_visible(False)
    ax8.spines[i].set_visible(False)
    ax9.spines[i].set_visible(False)

""">>>3.2.3. Distribusi fitur kategorikal terhadap variabel target"""

fig = plt.figure(figsize=(13,8))
gs = fig.add_gridspec(6,2)
gs.update(wspace=0.5, hspace=0.5)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])
ax4 = fig.add_subplot(gs[2,0])
ax5 = fig.add_subplot(gs[2,1])
ax6 = fig.add_subplot(gs[3,0])
ax7 = fig.add_subplot(gs[3,1])
ax8 = fig.add_subplot(gs[4,0])
ax9 = fig.add_subplot(gs[4,1])
ax10 = fig.add_subplot(gs[5,0])
ax11 = fig.add_subplot(gs[5,1])

color_palette = ["#800000","#8000ff","#6aac90","#5833ff","#da8829"]

# Cp
# 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic
ax0.text(0.5,0.5,"Distribusi\nchest pain\n__________",
        horizontalalignment = 'center', verticalalignment = 'center',
        fontsize = 12, fontweight='bold', fontfamily='serif', color='#000000')
ax0.spines["bottom"].set_visible(False)
ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.tick_params(left=False, bottom=False)
ax0.text(1,.5,"0 - Typical Angina\n1 - Atypical Angina\n2 - Non-anginal Pain\n3 - Asymptomatic",
        horizontalalignment = 'center', verticalalignment = 'center', fontsize = 10 )
ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax1, data=df, x='cp',hue="target", fill=True,palette=["#8000ff","#da8829"], alpha=.5, linewidth=0)
ax1.set_xlabel("")
ax1.set_ylabel("")

# Ca
ax2.text(0.5,0.5,"Jumlah of\npembuluh darah arteri\n___________",
        horizontalalignment = 'center', verticalalignment = 'center',
        fontsize = 12, fontweight='bold', fontfamily='serif', color='#000000')
ax2.text(1,.5,"0 vessels\n1 vessel\n2 vessels\n3 vessels\n4vessels",
        horizontalalignment = 'center', verticalalignment = 'center', fontsize = 10)
ax2.spines["bottom"].set_visible(False)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.tick_params(left=False, bottom=False)
ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax3, data=df, x='ca',hue="target", fill=True,palette=["#8000ff","#da8829"], alpha=.5, linewidth=0)
ax3.set_xlabel("")
ax3.set_ylabel("")

# Sex
ax4.text(0.5,0.5,"Serangan jantung\nberdasarkan\njenis kelamin\n______",
        horizontalalignment = 'center', verticalalignment = 'center',
        fontsize = 12, fontweight='bold', fontfamily='serif', color='#000000')
ax4.text(1,.5,"0 - Female\n1 - Male",
        horizontalalignment = 'center', verticalalignment = 'center', fontsize = 10)
ax4.spines["bottom"].set_visible(False)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.tick_params(left=False, bottom=False)
ax5.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax5,data=df,x='sex',palette=["#8000ff","#da8829"], hue='target')
ax5.set_xlabel("")
ax5.set_ylabel("")

# Thall
ax6.text(0.5,0.5,"Distribusi thal\nberdasarkan\n target variable\n___________",
        horizontalalignment = 'center', verticalalignment = 'center',
        fontsize = 12, fontweight='bold', fontfamily='serif', color='#000000')
ax6.text(1,.5,"Thalium Stress\nTest Result\n0, 1, 2, 3",
        horizontalalignment = 'center', verticalalignment = 'center', fontsize = 10)
ax6.spines["bottom"].set_visible(False)
ax6.set_xticklabels([])
ax6.set_yticklabels([])
ax6.tick_params(left=False, bottom=False)
ax7.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax7, data=df, x='thal',hue="target", fill=True,palette=["#8000ff","#da8829"], alpha=.5, linewidth=0)
ax7.set_xlabel("")
ax7.set_ylabel("")

# Thalach
ax8.text(0.5,0.5,"Boxen plot \nthalachh wrt\noutcome\n_______",
        horizontalalignment = 'center', verticalalignment = 'center',
        fontsize = 12, fontweight='bold', fontfamily='serif', color='#000000')
ax8.text(1,.5,"Maximum heart\nrate achieved",
        horizontalalignment = 'center', verticalalignment = 'center', fontsize = 10)
ax8.spines["bottom"].set_visible(False)
ax8.set_xticklabels([])
ax8.set_yticklabels([])
ax8.tick_params(left=False, bottom=False)
ax9.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.boxenplot(ax=ax9, data=df,x='target',y='thalach',palette=["#8000ff","#da8829"])
ax9.set_xlabel("")
ax9.set_ylabel("")

# Exang
ax10.text(0.5,0.5,"Strip Plot of\nexang vs age\n______",
        horizontalalignment = 'center', verticalalignment = 'center',
        fontsize = 12, fontweight='bold', fontfamily='serif', color='#000000')
ax10.text(1,.5,"Exercise induced\nangina\n0 - No\n1 - Yes",
        horizontalalignment = 'center', verticalalignment = 'center', fontsize = 10)
ax10.spines["bottom"].set_visible(False)
ax10.set_xticklabels([])
ax10.set_yticklabels([])
ax10.tick_params(left=False, bottom=False)
ax11.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.stripplot(ax=ax11, data=df,x='exang',y='age',hue='target',palette=["#8000ff","#da8829"])
ax9.set_xlabel("")
ax9.set_ylabel("")

for i in ["top","left","right"]:
    ax0.spines[i].set_visible(False)
    ax1.spines[i].set_visible(False)
    ax2.spines[i].set_visible(False)
    ax3.spines[i].set_visible(False)
    ax4.spines[i].set_visible(False)
    ax5.spines[i].set_visible(False)
    ax6.spines[i].set_visible(False)
    ax7.spines[i].set_visible(False)
    ax8.spines[i].set_visible(False)
    ax9.spines[i].set_visible(False)
    ax10.spines[i].set_visible(False)
    ax11.spines[i].set_visible(False)

""">>>3.2.4. Pairplot berdasarkan variabel target"""

sns.pairplot(df,hue='target',palette = ["#8000ff","#da8829"])
plt.show()

"""4. Data Preprocessing
> 4.1. Drop fitur dengan korelasi rendah
"""

df.drop(['restecg', 'fbs'], axis=1, inplace=True)
df.head()

df.shape

""">4.2. Train and test split"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, plot_importance

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score

cat_cols = ['sex','cp','ca','thal','exang','slope']
dfc = pd.get_dummies(df, columns=cat_cols)
dfc.head()

x = dfc.drop('target', axis=1)
y = dfc['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f'Total sampel seluruh dataset: {len(x)}')
print(f'Total sampel pada train dataset: {len(x_train)}')
print(f'Total sampel pada test dataset: {len(x_test)}')

print(f'Total sampel seluruh dataset: {len(y)}')
print(f'Total sampel pada train dataset: {len(y_train)}')
print(f'Total sampel pada test dataset: {len(y_test)}')

""">4.3. Standarisasi dengan menggunakan StandardScaler()"""

cont_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
sc = StandardScaler()
sc.fit(x_train[cont_cols])
x_train[cont_cols] = sc.transform(x_train.loc[:, cont_cols])
x_test[cont_cols] = sc.transform(x_test.loc[:, cont_cols])
x_train[cont_cols].head()

x_train[cont_cols].describe().round(4)

"""5. Modelling
>5.1. Random Forest
"""

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf.fit(x_train, y_train)
rf_predicted = rf.predict(x_test)
rf_acc_score = accuracy_score(y_test, rf_predicted)

print(classification_report(y_test, rf_predicted))
print("Accuracy of Random Forest: {:.2f}%\n".format(round(rf_acc_score*100, 2)))

""">5.2. Decision Tree"""

dt = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=7)
dt.fit(x_train, y_train)
dt_predicted = dt.predict(x_test)
dt_acc_score = accuracy_score(y_test, dt_predicted)

print(classification_report(y_test, dt_predicted))
print("Accuracy of Decision Tree Classifier: {:.2f}%\n".format(round(dt_acc_score*100, 2)))

""">5.3. Support Vector Machine (SVM)"""

svc =  SVC(kernel='rbf', C=2)
svc.fit(x_train, y_train)
svc_predicted = svc.predict(x_test)
svc_acc_score = accuracy_score(y_test, svc_predicted)

print(classification_report(y_test,svc_predicted))
print("Accuracy of Support Vector Classifier: {:.2f}%\n".format(round(svc_acc_score*100, 2)))

"""Extreme Gradient Boosting (XGBoost)"""

xgb = XGBClassifier(learning_rate=0.01, n_estimators=100, max_depth=15)
xgb.fit(x_train, y_train)
xgb_predicted = xgb.predict(x_test)
xgb_acc_score = accuracy_score(y_test, xgb_predicted)

print(classification_report(y_test, xgb_predicted))
print("Accuracy of Extreme Gradient Boost: {:.2f}%\n".format(round(xgb_acc_score*100, 2)))

"""6. Evaluasi Model"""

eval_mod = pd.DataFrame({'Model': ['Random Forest', 'Decision Tree', 'SVM', 'XGBoost'],
                         'Akurasi': [rf_acc_score*100, dt_acc_score*100, svc_acc_score*100, xgb_acc_score*100]})
eval_mod

colors = ['blue', 'green', 'red', 'yellow']
plt.figure(figsize=(8,3))
plt.title("Barplot Akurasi Model")
plt.xlabel("Algoritma")
plt.ylabel("Accuracy %")
plt.bar(eval_mod['Model'],eval_mod['Akurasi'],color = colors)
plt.show()

rf_precision = precision_score(y_test,rf_predicted)
rf_recall = recall_score(y_test,rf_predicted)
rf_f1 = f1_score(y_test,rf_predicted)

dt_precision = precision_score(y_test,dt_predicted)
dt_recall = recall_score(y_test,dt_predicted)
dt_f1 = f1_score(y_test,dt_predicted)

svc_precision = precision_score(y_test,svc_predicted)
svc_recall = recall_score(y_test,svc_predicted)
svc_f1 = f1_score(y_test,svc_predicted)

xgb_precision = precision_score(y_test,xgb_predicted)
xgb_recall = recall_score(y_test,xgb_predicted)
xgb_f1 = f1_score(y_test,xgb_predicted)

eval_mod['Precision'] = [rf_precision, dt_precision, svc_precision, xgb_precision]
eval_mod['Recall'] = [rf_recall, dt_recall, svc_recall, xgb_recall]
eval_mod['F1-score'] = [rf_f1, dt_f1, svc_f1, xgb_f1]
print(eval_mod)

rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(y_test,rf_predicted)
dt_false_positive_rate,dt_true_positive_rate,dt_threshold = roc_curve(y_test,dt_predicted)
svc_false_positive_rate,svc_true_positive_rate,svc_threshold = roc_curve(y_test,svc_predicted)
xgb_false_positive_rate,xgb_true_positive_rate,xgb_threshold = roc_curve(y_test,xgb_predicted)

rf_roc_auc = roc_auc_score(y_test, rf_predicted)
dt_roc_auc = roc_auc_score(y_test, dt_predicted)
svc_roc_auc = roc_auc_score(y_test, svc_predicted)
xgb_roc_auc = roc_auc_score(y_test, xgb_predicted)

plt.figure(figsize=(10,5))
plt.title('Receiver Operating Characteristic Curve')
plt.plot(rf_false_positive_rate,rf_true_positive_rate,
         label='Random Forest (area = %0.3f)' % rf_roc_auc)
plt.plot(dt_false_positive_rate,dt_true_positive_rate,
         label='Decision Tree (area = %0.3f)' % dt_roc_auc)
plt.plot(svc_false_positive_rate,svc_true_positive_rate,
         label='Support Vector Classifier (area = %0.3f)' % svc_roc_auc)
plt.plot(xgb_false_positive_rate,xgb_true_positive_rate,
         label='Extreme Gradient Boost (area = %0.3f)' % xgb_roc_auc)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.4')
plt.plot([1,1],c='.4')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()

#mencari fitur yang paling berpengaruh
reg = XGBClassifier().fit(x, y)

# plot the feature importance
plot_importance(reg)
plt.show()

