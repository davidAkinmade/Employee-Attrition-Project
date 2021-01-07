# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 13:01:01 2020

@author: akinmade
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#importing the datasets from the csv files
df1 = pd.read_excel('problem case.xlsx', sheet_name='Existing employees')
df2 = pd.read_excel('problem case.xlsx', sheet_name='Employees who have left')

#checking for null/missing variables
df1.isnull().sum()
df2.isnull().sum()
#combining both datasets into one for analysis
df3 = pd.concat([df1, df2])

df3.columns

#carrying out univariate analysis
df3_uni = df3[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'dept', 'salary', 'attrition']]

#plotting histogram for the columns provided in the dataset
for i in df3_uni.columns:
    cat_num = df3_uni[i].value_counts()
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.show()
    
#carrying out bivariate analysis with a heatmap to check correlation
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(df3.corr(),vmax=.3, center=0, cmap=cmap,
            square=True, linewidths=.5)
    
#bivariate analysis between attrition and non_categorical features
df3_non_cat = df3.drop(['dept', 'salary', 'attrition'], axis=1)
for column in df3_non_cat:
    df_plot = pd.melt(df3, id_vars='attrition', value_vars=column, value_name='value')
    bins=np.linspace(df3_non_cat[column].min(), df3_non_cat[column].max(), 5)
    g = sns.FacetGrid(df_plot, col="variable", hue="attrition", palette="Set1", col_wrap=2)
    plt.xticks(rotation=90)
    g.map(plt.hist, 'value', bins=bins, ec="k")

    g.axes[-1].legend()

    plt.show()
   
#bivariate analysis between attrition and categorical features
df3_cat = df3[['dept', 'salary']]    
for column in df3_cat:
    df_plot = pd.melt(df3, id_vars='attrition', value_vars=column, value_name='value')
    g = sns.FacetGrid(df_plot, col="variable", hue="attrition", palette="Set1", col_wrap=2)
    plt.xticks(rotation=90)
    g.map(plt.hist, 'value', ec="k")

    g.axes[-1].legend()

    plt.show()
    
#exporting our explored and cleaned dataset to a csv file
df3.to_csv('problem case combined.csv', index=False)  
    
    
    
    
    
    
    
    
    
    
    