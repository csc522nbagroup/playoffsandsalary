# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 15:36:14 2018

@author: chris
"""

from __future__ import division
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
dff=pd.read_csv('original.csv')

df = pd.read_csv('original.csv')

df['Playoffs'] = (df['Playoffs']=='Y')+0 # Convert Y/N to 0/1

# Drop categorical columns
droplist = []
for a,b in zip(list(df),df.dtypes):
    print("a=",a,"b=",b)
    if b==object:
        droplist.append(a)

print(droplist)        

print("Before rename and drop")
print(list(df))

df = df.drop(droplist,1)

df.columns = df.columns.str.replace(' ','_')
df.columns = df.columns.str.replace('\\n','_')
df.columns = df.columns.str.replace('.','_')
df.columns = df.columns.str.replace('-','dash')
df.columns = df.columns.str.replace('+','plus')

df.columns = df.columns.str.replace('%','_pct_')
df.columns = df.columns.str.replace('/','_p_')
feature=['Contrib', 'Dcontrib', 'DVORP', 'Ocontrib', 'BPM_2', 'VORP_1', 'DdashBPM', 'OBPM_1', 'OVORP', 'RAW_SPM', 'WS_p_48', 'Tm_TS_W_p_O_Plyr', 'Team_Mar', 'Team_MP', 'Team_TS_pct_', 'Year', 'BPM_1', 'OdashBPM', 'Raw_OBPM', 'Team_Gm', 'Age_on_Feb_1', 'BPM', 'WORP', 'StdErr', 'Sum_SPM', 'ReMPG', 'WS', 'Offense', 'TRB_pct_', 'DStdErr', 'Adjusted_WORP', 'OdashWORP', 'Height', 'Shot_pct_', 'VORP_2', 'VORPdashGm', 'MPG_plus_Int', 'DdashWORP', 'TS_pct_', 'ORB_pct_', 'Tm_Ortg', 'OBPM', 'Contrib_2', 'OVORP_1', 'Age', 'STL_pct_', 'DBPM_1', 'DVORP_1', 'Exp_BPM', 'Estimated_Position', 'BLK_pct_', 'Weight', 'PER', 'MPG_1', 'MP', 'TrueTalent_VORP', 'Rounded_Age', 'Contrib_1', 'Val_p_Shot_1', 'TrueTalent_BPM', 'Tm_Adj', 'Tm_USG', 'Rounded_Position', 'VORP', 'USG_pct_', 'Ocontrib_1', '_pct_Min_1', 'Offense_1', 'DWS', 'AST_pct_', 'OVORP_Gm', 'Exp_pct__Min', 'BBRef_Pos', 'DVORP_Gm', 'TrueTimeVORP', 'DRB_pct_']


df = df.dropna()
df=df.drop(feature,axis=1)
new_data=df
new_data.to_csv('after.csv')
print(list(df))

