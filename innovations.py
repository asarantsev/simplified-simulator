import numpy as np
import pandas as pd
from statsmodels.api import OLS

# This is a technical command needed to write data in pandas data frames
pd.options.mode.chained_assignment = None

# read the innovations Excel file
dataDF = pd.read_excel('innovations.xlsx')

# and add a constant series for regressions
dataDF['const'] = 1

# this is the function which fills the missing data using linear regression
# with ordinary least squares and picking a residual at random uniformly
# DF is the data frame which we fill
# fullKeys is the list of all keys for variables with full data
# missingKey is the key for the variable with missing data
# nmiss is the number of missing data points
# they are presumed to be at the end of the data frame
def fill(DF, fullKeys, missingKey, nmiss):
    regDF = DF[fullKeys] # the series with full data
    # Regression of short series upon full series with missing data
    Reg = OLS(DF[missingKey].iloc[:-nmiss], regDF.iloc[:-nmiss]).fit()
    res = Reg.resid.values # residuals
    # sample with replacement these residuals
    pick = np.random.choice(res, nmiss)
    # for each missing data point
    # add regression-predicted value to sampled residual
    # write them in separate list called 'missing'
    missing = [Reg.predict(regDF.iloc[k-nmiss]) + pick[k] for k in range(nmiss)]
    outputDF = DF # create a copy of original data frame
    for k in range(nmiss):
        # and write imputed data in the correct place
        outputDF[missingKey].iloc[k-nmiss] = missing[k]
    return outputDF

# fill the one missing data point for log volatility
DF1 = fill(dataDF, ['spread', 'baa-rate', 'earn-growth', 'usa-ret', 'const'], 'vol', 1)

# fill data points for international stock returns
DF2 = fill(DF1, ['vol', 'spread', 'baa-rate', 'earn-growth', 'usa-ret', 'const'], 'intl-ret', 97-55)

# fill data points for corporate bond returns
DF3 = fill(DF2, ['vol', 'spread', 'baa-rate', 'earn-growth', 'usa-ret', 'intl-ret', 'const'], 'bond-ret', 97-52)

# write this filled dataframe in the excel file
# do not forget to drop the constant series
# and manually delete row numbers in the Excel file
DF3[['vol', 'spread', 'baa-rate', 'earn-growth', 'usa-ret', 'intl-ret', 'bond-ret']].to_excel('filled.xlsx')