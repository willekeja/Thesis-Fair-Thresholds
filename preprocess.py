

import pandas as pd
import numpy as np
import sklearn.preprocessing



def pre_process_credit(df):

    df.columns = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount', 
         'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors', 
         'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing', 
         'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'labels']


    numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age', 
           'existingcredits', 'peopleliable', 'labels']

    df[numvars] = df[numvars].apply(lambda x: x.astype(float))

    labels = df['labels']

    df.drop("labels", axis=1, inplace=True)

    bins = [-1, 25, 99999] # recode age as categorical variable with values from A comparative study of fairness-enhancing interventions in machine learning by Friedler et. al.

    df.age = pd.cut(df.age, bins=bins, labels= ['young', 'old'], right = False)

    catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
           'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job', 
           'telephone', 'foreignworker', 'age']
    
    df = pd.get_dummies(df, drop_first = True, columns=catvars)
    
    colnames = df.columns 

    scaler = sklearn.preprocessing.MinMaxScaler()

    df = scaler.fit_transform(df)

    df = pd.DataFrame(df, columns = colnames)

    df['label'] = labels-1
    
    df.name = 'credit'
    
    df.baseline = 0.7
    
    return(df)


def pre_process_compas(df):
    
    vars_to_drop = ['Unnamed: 0', 'age_cat', 'sex', 'score_text', 'is_recid', 'score_factor', 'race', 'age_factor', 'c_offense_date']

    df.drop(vars_to_drop, axis=1, inplace=True,)
    
    numvars = ['age', 'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'two_year_recid', 'decile_score', 'in_custody',
               'out_custody', 'days_b_screening_arrest', 'days_b_screening_arrest', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']

    df[numvars] = df[numvars].apply(lambda x: x.astype(float))


    catvars = ["c_charge_degree", "race_factor", 'gender_factor', 'crime_factor']
        
    df = pd.get_dummies(df, drop_first = True, columns = catvars)

    labels = df["two_year_recid"].values
    
    df.drop("two_year_recid", axis=1, inplace=True)
    
    colnames = df.columns 

    scaler = sklearn.preprocessing.MinMaxScaler()

    df = scaler.fit_transform(df)

    df = pd.DataFrame(df, columns = colnames)

    df['label'] = labels
    
    df.name = 'compas'
    
    df.baseline = 0.54

    return(df)
  
def pre_process_adult(df):
  
    
    df = df.replace('?', np.nan)
    
    df = df.dropna()

    df.columns = ["Age", "WorkClass", "fnlwgt", "Education", "EducationNum", "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
                  "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"]

    df["Income"] = df["Income"].map({ "<=50K": 0, ">50K": 1 }) # create binary variable with values from literature
    
    labels = df["Income"].values
    
    vars_to_drop = ["fnlwgt", "Education", "Income"]
    
    df.drop(vars_to_drop, axis=1, inplace=True,)

    numvars = ['CapitalGain', 'CapitalLoss', 'Age', 'EducationNum', 'HoursPerWeek']
    
    df[numvars] = df[numvars].apply(lambda x: x.astype(float))
    
    catvars =  ["WorkClass", "MaritalStatus", "Occupation", "Relationship",
    "Race", "Gender", "NativeCountry"]
        
    df = pd.get_dummies(df, drop_first = True, columns = catvars)
    
    colnames = df.columns 

    scaler = sklearn.preprocessing.MinMaxScaler()

    df = scaler.fit_transform(df)

    df = pd.DataFrame(df, columns = colnames)

    df['label'] = labels
    
    df.name = 'adult'
    
    df.baseline = 0.75
     
    return(df)  


def pre_process_student(df):

  df["G3"] = pd.cut(df["G3"], bins=[0, 10, 21], right = False, duplicates = 'drop', labels = False)
    
  numeric_columns = ['age', 'Medu', 'Fedu', 'famrel', 'traveltime', 'studytime', 'failures', 'freetime', 'goout', 'Walc', 'Dalc', 'health']

  df[numeric_columns] = df[numeric_columns].apply(lambda x: x.astype(float))


  labels = df.G3
  
  vars_to_drop = ["G3", 'G1', 'G2', 'absences'] 
  
  df.drop(vars_to_drop, axis=1, inplace=True,)
    
  cat_columns = [i for i in df.columns if i not in numeric_columns]

  for col in cat_columns:
    df[col] = pd.Categorical(df[col], categories=df[col].unique()).codes



  df = pd.get_dummies(df, drop_first = True, columns = cat_columns)

  colnames = df.columns 

  scaler = sklearn.preprocessing.MinMaxScaler()

  df = scaler.fit_transform(df)

  df = pd.DataFrame(df, columns = colnames)

  df['label'] = labels

  df.name = 'student'

def pre_process_large_credit(df):
  
  df = df.dropna()
  
  df.MARRIAGE  = pd.cut(df["MARRIAGE"], bins=[0, 2, 4], right = False, duplicates = 'drop', labels = False)

  cat_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'default payment next month']
  
  
  for col in cat_columns:
    df[col] = pd.Categorical(df[col], categories=df[col].unique()).codes

  numeric_columns = [i for i in df.columns if i not in cat_columns]
  

  vars_to_drop = ['default payment next month']
  labels = df['default payment next month']

  df.drop(vars_to_drop, axis=1, inplace=True)
  
  cat_columns = ['SEX', 'EDUCATION', 'MARRIAGE']

  df = pd.get_dummies(df, drop_first = True, columns = cat_columns)

  colnames = df.columns 
  
  scaler = sklearn.preprocessing.MinMaxScaler()

  df = scaler.fit_transform(df)


  df = pd.DataFrame(df, columns = colnames)

  df['label'] = labels

  df.name = 'large_credit'
  
  df =  df.dropna()
  
  return(df)


def pre_process_churn(df):
  
  df = df.iloc[:,1:]

  df = df.replace(' ', np.nan)

  df = df.dropna()

  numeric_columns = ['MonthlyCharges', 'TotalCharges', 'tenure']

  df[numeric_columns] = df[numeric_columns].apply(lambda x: x.astype(float))
  
  df["Churn"] = df["Churn"].map({ "No": 0, "Yes": 1 }) # create binary variable with values from literature
 
  labels = df.Churn

  vars_to_drop = ['Churn']

  df.drop(vars_to_drop, axis=1, inplace=True,)

  cat_columns = [i for i in df.columns if i not in numeric_columns]

  for col in cat_columns:
    df[col] = pd.Categorical(df[col], categories=df[col].unique()).codes


  df = pd.get_dummies(df, drop_first = True, columns = cat_columns)
  
  df = df.dropna()

  colnames = df.columns 
  
  scaler = sklearn.preprocessing.MinMaxScaler()

  df = scaler.fit_transform(df)

  df = pd.DataFrame(df, columns = colnames)

  df['label'] = labels
    
  df.name = 'churn'
    
  return(df)

def pre_process_reincidencia(df):

  df = df.dropna(axis = 'columns')

  cat_columns = ['V132_REINCIDENCIA_2013','V122_rein_fet_2013', 'V115_reincidencia_2015', 'V27_durada_programa_agrupat', 'V25_programa_mesura', 'V24_programa', 'V23_territori','V21_fet_nombre', 'V19_fets_desagrupats',
  'V17_fet_tipus', 'V16_fet_violencia', 'V15_fet_agrupat', 'V14_fet', 'V13_nombre_fets_agrupat', 'V11_antecedents', 'V7_comarca', 'V3_nacionalitat', 'V2_estranger',
  'V1_sexe']
  
  for col in cat_columns:
   
    df[col] = pd.Categorical(df[col], categories=df[col].unique()).codes
      
  labels = df.V115_reincidencia_2015
  
  df.program_duration = ((pd.to_datetime(df.V31_data_fi_programa,infer_datetime_format=True) - pd.to_datetime(df.V30_data_inici_programa,infer_datetime_format=True))/np.timedelta64(1, 's')).astype(float)
  
  vars_to_drop = [ 'V122_rein_fet_2013', 'V31_data_fi_programa' , 'V30_data_inici_programa', 'V10_data_naixement', 'V22_data_fet', 'V132_REINCIDENCIA_2013', 'V115_reincidencia_2015']
  
  df.age = pd.to_datetime(df.V10_data_naixement,infer_datetime_format=True).to_numpy().astype('datetime64[Y]').astype(float) + 1970
  
  df.drop(vars_to_drop, axis=1, inplace=True)
  
  cat_columns = ['V19_fets_desagrupats', 'V21_fet_nombre','V27_durada_programa_agrupat', 'V14_fet', 'V25_programa_mesura', 'V24_programa', 'V23_territori',
  'V17_fet_tipus', 'V16_fet_violencia', 'V15_fet_agrupat', 'V13_nombre_fets_agrupat', 'V11_antecedents', 'V7_comarca', 'V3_nacionalitat', 'V2_estranger',
  'V1_sexe']
  
  df = pd.get_dummies(df, drop_first = True, columns = cat_columns)
   
  colnames = df.columns 
  
  scaler = sklearn.preprocessing.MinMaxScaler()
 
  df = scaler.fit_transform(df)
 
  df = pd.DataFrame(df, columns = colnames)
  
  df['label'] = labels
  
  df.name = 'reincidencia'
  
  return(df)