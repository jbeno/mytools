# mytools
Some helpful python functions for data science. I'm not a software engineer, be kind!

## get_unique
Gets all the unique values of variables in a dataframe. For each variable, if the number of unique values is equal to or below "n", it will treat them as categorical and output the unique values (optionally with counts, percentages). For variables with unique values above "n", it will consider them continuous numerical. If "cont" is True, you will see a statistical summary of these. If "plot" is True, charts will be drawn for all of the variables (categorical will get bar charts, continuous will get histograms).
I created this because I was tired of fetching all the unique values by hand. It's meant to speed up initial exploration of a dataset.
```
def get_unique(df, n=20, sort='none', list=True, strip=False, count=False, percent=False, plot=False, cont=False):

    Parameters:
    - df: dataframe that contains the variables you want to analyze
    - n: int (default is 20). Maximum number of unique values to consider (avoid iterating continuous data)
    - sort: str, optional (default='none'). Determines the sorting of unique values:
        'none' will keep original order,
        'name' will sort alphabetically/numerically,
        'count' will sort by count of unique values (descending)
    - list: boolean, optional (default=True). Shows the list of unique values
    - strip: boolean, optional (default=False). True will remove single quotes in the variable names
    - count: boolean, optional (default=False). True will show counts of each unique value
    - percent: boolean, optional (default=False). True will show percentage of each unique value
    - plot: boolean, optional (default=False). True will show a basic chart for each variable
    - cont: boolean, optional (default=False). True will analyze variables over n as continuous
    
    Returns: None
```
<img src="/images/get_unique_1.png" width="300" valign="top"> <img src="/images/get_unique_2.png" width="300" valign="top"> <img src="/images/get_unique_3.png" width="300" valign="top">

## plot_charts
Gets all the variables in a dataframe and outputs a grid of charts. For each variable, if the number of unique values is equal to or below "n", it will treat them as categorical and plot a bar chart. For variables with unique values above "n", it will consider them continuous and plot a histograms.
I created this because I was tired of plotting these charts one at a time during the very first stage of data visualization. Pairplots work too, but the histograms are tiny, and this is meant for early exploration before you start comparing variables. 
```
def plot_charts(df, n=10, ncols=3, figsize=(20, 40), rotation=45):

    Plot histograms for each column in a DataFrame in a grid of subplots.

    Parameters:
    - df: dataframe that contains the variables you want to analyze
    - n: int (default=20). Threshold of unique values for categorical (equal or below) vs. continuous (above)
    - ncols: int, optional (default=3). The number of columns in the subplot grid.
    - figsize: tuple of ints, optional (default=(20, 20)). The size of the entire plot figure.
    - rotation: int, optional (default=45). The rotation of the x-axis labels.

    Returns: None
```

<img src="/images/plot_charts.png" width="400">

## split_dataframe
Take a dataframe as input, along with "n", and splits all variables below "n" into a categorical dataframe (df_cat), and the others in a continuous numerical dataframe (df_num).
This was meant to help isolate the sets of variables you'd be using for correlations vs. categorical explorations.
```
def split_dataframe(df, n):

    Split a DataFrame into two based on the number of unique values in each column.

    Parameters:
    - df: DataFrame. The DataFrame to split.
    - n: int. The maximum number of unique values for a column to be considered categorical.

    Returns:
    - df_cat: DataFrame. Contains the columns of df with n or fewer unique values.
    - df_num: DataFrame. Contains the columns of df with more than n unique values.
```

Hopefully you find some of these helpful. Feel free to improve and share!


