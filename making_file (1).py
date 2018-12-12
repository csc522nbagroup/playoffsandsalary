# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 23:37:01 2018

@author: wangh
"""

import pandas as pd
wsh=pd.read_csv('original.csv')
wsh=wsh.dropna()
features=['Contrib', 'Dcontrib', 'DVORP', 'Ocontrib', 'BPM_2', 'VORP_1', 'DdashBPM', 'OBPM_1', 'OVORP', 'const', 'RAW_SPM', 'WS_p_48', 'Tm_TS_W_p_O_Plyr', 'Team_Mar', 'Team_MP', 'Team_TS_pct_', 'Year', 'BPM_1', 'OdashBPM', 'Raw_OBPM', 'Team_Gm', 'Age_on_Feb_1', 'BPM', 'WORP', 'StdErr', 'Sum_SPM', 'ReMPG', 'WS', 'Offense', 'TRB_pct_', 'DStdErr', 'Adjusted_WORP', 'OdashWORP', 'Height', 'Shot_pct_', 'VORP_2', 'VORPdashGm', 'MPG_plus_Int', 'DdashWORP', 'TS_pct_', 'ORB_pct_', 'Tm_Ortg', 'OBPM', 'Contrib_2', 'OVORP_1', 'Age', 'STL_pct_', 'DBPM_1', 'DVORP_1', 'Exp_BPM', 'Estimated_Position', 'BLK_pct_', 'Weight', 'PER', 'MPG_1', 'MP', 'TrueTalent_VORP', 'Rounded_Age', 'Contrib_1', 'Val_p_Shot_1', 'TrueTalent_BPM', 'Tm_Adj', 'Tm_USG', 'Rounded_Position', 'VORP', 'USG_pct_', 'Ocontrib_1', '_pct_Min_1', 'Offense_1', 'DWS', 'AST_pct_', 'OVORP_Gm', 'Exp_pct__Min', 'BBRef_Pos', 'DVORP_Gm', 'TrueTimeVORP', 'DRB_pct_']
wsh.drop(features,axis=1)