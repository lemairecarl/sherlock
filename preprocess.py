import numpy as np
import pandas as pd
import torch
from scipy.cluster.vq import whiten

print('Loading data...')
dataf = pd.read_csv('hochelaga.csv')

print('Removing duplicates...')
print('Before drop dupl', len(dataf))
dataf = dataf.drop_duplicates('code_postal')
print(' After drop dupl', len(dataf))

target_col_names = ['pct_qs',
                    'pct_pq',
                    'pct_plq',
                    'pct_caq',
                    'pct_on',
                    'pct_pv']
target_vars = [dataf[k] for k in target_col_names]

print('Selecting and transforming variables...')
sel_vars = [
    dataf['lat'],
    dataf['lon'],
    #dataf['zone_urbaine'],
    dataf['appartement?'],
    # POPULATION
    dataf['PP_FEMALE'] / dataf['PP_TOT'],
    dataf['PP_PP_MED'],
    dataf['PP_LMR'] / dataf['PP_POP15_'],  # PP_LMR/PP_POP15_
    ##FAMILY
    dataf['FM_2P'] / dataf['FM_TOT'],  #"m_FM_2_FAM_PROP" = FM_2P/FM_TOT,
    dataf['FM_LP'] / dataf['FM_TOT'],  #"m_FM_LONE_PARENT_PROP" = FM_LP/FM_TOT,
    dataf['FM_AVCH'], #"m_FM_KIDS_AVG" = FM_AVCH,
    dataf['FM_AVG_P'], #"m_FM_PERSON_AVG" = FM_AVG_P,
    ##HOUSEHOLDS
    dataf['HH_RENT'] / dataf['HH_TOT'],  #"m_HH_RENT_PROP" = HH_RENT/HH_TOT,
    ##DWELLING
    dataf['DW_SINGLE'] / dataf['DW_TOT'],  #"m_DW_SINGLE_PROP" = DW_SINGLE/DW_TOT,
    dataf['DW_MVALUE'],  #"m_DW_VALUE_MED" = DW_MVALUE,
    dataf['DW_ARMSPDW'],  #"m_DW_ROOM_AVG" = DW_ARMSPDW,
    (dataf['DW_CON0611'] + dataf['DW_CON12_']) / dataf['DW_TOT'],  #"m_DW_CON12_PLUS_PROP" =  (DW_CON0611+DW_CON12_)/DW_TOT,
    dataf['DW_PTCON'] / dataf['DW_TOT'],  #"m_DW_CONDO_PROP" = DW_PTCON/DW_TOT,
    ##EDUCATION - Could be interesting to add more fields of study
    dataf['ED_15UD'] / dataf['ED_15HL'],  #"m_ED_UNI_PROP" = ED_15UD / ED_15HL,
    dataf['ED_MJADM'] / dataf['ED_MJ'],  #"m_ED_BUSINESS_PROP" =  ED_MJADM / ED_MJ,
    dataf['ED_MJJART'] / dataf['ED_MJ'],  #"m_ED_ART_PROP" = ED_MJJART / ED_MJ,
    ##MINORITY
    dataf['MN_VIS'] / dataf['MN_TOT'],  #"m_MN_MIN_VIS_PROP" = MN_VIS/MN_TOT,
    ##IMMIGRATION
    dataf['IM_CNTRY'] / dataf['IM_POP'],  #"m_IM_IM_PROP" =  IM_CNTRY/IM_POP,
    dataf['IM_GEN3'] / dataf['IM_TOTGEN'],  #"m_IM_3RD_GEN_PROP" = IM_GEN3/IM_TOTGEN,
    dataf['IM_SFR'] / dataf['IM_POP'],  #"m_IM_FRENCH_PROP" = IM_SFR/IM_POP,
    ##OFFICIAL LANGUAGE
    dataf['OL_F_ON'] / dataf['OL_TOT'],  #"m_OL_FRE_PROP" = OL_F_ON / OL_TOT,
    dataf['OL_E_ON'] / dataf['OL_TOT'],  #"m_OL_ENG_PROP" = OL_E_ON / OL_TOT,
    dataf['OL_NOEF'] / dataf['OL_TOT'],  #"m_OL_N_ENG_FRE_PROP" =  OL_NOEF / OL_TOT,
    ##HOME LANGUAGE - Something to do with home language?
    #
    ##MOBILITY
    dataf['MB_5YNM'] / dataf['MB_5YTOT'],  #"m_MB_NON_MOVER_PROP" = MB_5YNM  / MB_5YTOT,
    ##OCCUPATION - Should be highly correlated with education
    #
    ##LABOR FORCE
    dataf['LF_15UEM'] / dataf['LF_15'],  #"m_LF_UNEMP_PROP" =  LF_15UEM / LF_15,
    ##EMPLOYMENT
    dataf['EM_PSM'] / dataf['EM_ALL'],  #"m_EM_SELF_PROP" =  EM_PSM / EM_ALL,
    (dataf['EM_TRCD'] + dataf['EM_TRCP']) / dataf['EM_TRTOT'],  #"m_EM_CAR_PROP" =  (EM_TRCD + EM_TRCP) / EM_TRTOT,
    dataf['EM_LGNOF'] / dataf['EM_LGTOT'],  #"m_EM_NON_OL_PROP" =   EM_LGNOF / EM_LGTOT,
    ##INCOME
    dataf['IN_PM15_'],  #"m_IN_PERSON_IN_MED" =   IN_PM15_,
    (dataf['IN_P_05'] + dataf['IN_P0510'] + dataf['IN_P1015'] + dataf['IN_P1520']) / dataf['IN_W15_'],  #"m_IN_UNDER_20K_PROP" =  (IN_P_05 + IN_P0510 + IN_P1015 + #IN_P1520) / IN_W15_,
    ##RELIGION
    dataf['RL_CHRI'] / dataf['RL_TOT'],  #"m_RL_CHRIST_PROP" =  RL_CHRI / RL_TOT,
    dataf['RL_NON'] / dataf['RL_TOT'],  #"m_RL_NO_REL_PROP" = RL_NON / RL_TOT,
    dataf['RL_JEW'] / dataf['RL_TOT'],  #"m_RL_JEW_PROP" = RL_JEW / RL_TOT,
    dataf['RL_MUSL'] / dataf['RL_TOT'],  #"m_RL_MUSL_PROP" = RL_MUSL / RL_TOT)
    ]

print('Converting to tensors...')
predictors = np.array([col.values for col in sel_vars]).T
predictors = np.nan_to_num(predictors)
#print(np.std(predictors, axis=0))
predictors -= np.mean(predictors, axis=0)
predictors = whiten(predictors)
targets = np.array([col.values for col in target_vars]).T
targets = targets / 100.0
predictors, targets = torch.Tensor(predictors), torch.Tensor(targets)

print('Predictiors:', predictors.shape)
print('Targets:    ', targets.shape)

#print('Normalizing input variables...')
#predictors -= torch.mean(predictors, dim=0)
#predictors /= torch.std(predictors, dim=0)

print('Saving...')
torch.save((predictors, targets), 'hochelaga_postal.pt')

print('Done.')
print('DELETER le valsplit!!!')
