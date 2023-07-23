import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from math import ceil


def get_unique(df, n=20, sort='none', list=True, strip=False, count=False, percent=False, plot=False, cont=False):
    """
    Version 0.2
    Obtains unique values of all variables below a threshold number "n", and can display counts or percents
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
    """
    # Calculate # of unique values for each variable in the dataframe
    var_list = df.nunique(axis=0)

    # Iterate through each categorical variable in the list below n
    print(f"\nCATEGORICAL: Variables with unique values equal to or below: {n}")
    for i in range(len(var_list)):
        var_name = var_list.index[i]
        unique_count = var_list[i]

        # If unique value count is less than n, get the list of values, counts, percentages
        if unique_count <= n:
            number = df[var_name].value_counts(dropna=False)
            perc = round(number / df.shape[0] * 100, 2)
            # Copy the index to a column
            orig = number.index
            # Strip out the single quotes
            name = [str(n) for n in number.index]
            name = [n.strip('\'') for n in name]
            # Store everything in dataframe uv for consistent access and sorting
            uv = pd.DataFrame({'orig': orig, 'name': name, 'number': number, 'perc': perc})

            # Sort the unique values by name or count, if specified
            if sort == 'name':
                uv = uv.sort_values(by='name', ascending=True)
            elif sort == 'count':
                uv = uv.sort_values(by='number', ascending=False)
            elif sort == 'percent':
                uv = uv.sort_values(by='perc', ascending=False)

            # Print out the list of unique values for each variable
            if list:
                print(f"\n{var_name} has {unique_count} unique values:\n")
                for w, x, y, z in uv.itertuples(index=False):
                    # Decide on to use stripped name or not
                    if strip:
                        w = x
                    # Put some spacing after the value names for readability
                    w_str = str(w)
                    w_pad_size = uv.name.str.len().max() + 7
                    w_pad = w_str + " " * (w_pad_size - len(w_str))
                    y_str = str(y)
                    y_pad_max = uv.number.max()
                    y_pad_max_str = str(y_pad_max)
                    y_pad_size = len(y_pad_max_str) + 3
                    y_pad = y_str + " " * (y_pad_size - len(y_str))
                    if count and percent:
                        print("\t" + str(w_pad) + str(y_pad) + str(z) + "%")
                    elif count:
                        print("\t" + str(w_pad) + str(y))
                    elif percent:
                        print("\t" + str(w_pad) + str(z) + "%")
                    else:
                        print("\t" + str(w))

            # Plot countplot if plot=True
            if plot:
                print("\n")
                if strip:
                    if sort == 'count':
                        sns.barplot(data=uv, x='name', y='number', order=uv.sort_values('number', ascending=False).name)
                    else:
                        sns.barplot(data=uv, x=uv.loc[0], y='number', order=uv.sort_values('name', ascending=True).name)
                else:
                    if sort == 'count':
                        sns.barplot(data=uv, x='orig', y='number', order=uv.sort_values('number', ascending=False).orig)
                    else:
                        sns.barplot(data=uv, x='orig', y='number', order=uv.sort_values('orig', ascending=True).orig)
                plt.title(var_name)
                plt.xlabel('')
                plt.ylabel('')
                plt.xticks(rotation=45)
                plt.show()

    if cont:
        # Iterate through each categorical variable in the list below n
        print(f"\nCONTINUOUS: Variables with unique values greater than: {n}")
        for i in range(len(var_list)):
            var_name = var_list.index[i]
            unique_count = var_list[i]

            if unique_count > n:
                print(f"\n{var_name} has {unique_count} unique values:\n")
                print(var_name)
                print(df[var_name].describe())

                # Plot countplot if plot=True
                if plot:
                    print("\n")
                    sns.histplot(data=df, x=var_name)
                    # plt.title(var_name)
                    # plt.xlabel('')
                    # plt.ylabel('')
                    # plt.xticks(rotation=45)
                    plt.show()


def plot_charts(df, plot_type='both', n=10, ncols=3, fig_width=20, subplot_height=4, rotation=45, strip=False,
                cat_cols=None, cont_cols=None, dtype_check=True, sample_size=None):
    """
    Version 0.2
    Plot barplots for categorical columns, or histograms for continuous columns, in a grid of subplots.

    Parameters:
    - df: dataframe that contains the variables you want to analyze
    - plot_type: string, optional (default='both'). Type of charts to plot: 'cat' for categorical, 'cont' for
        continuous, 'both' for both
    - n: int (default=20). Threshold of unique values for categorical (equal or below) vs. continuous (above)
    - ncols: int, optional (default=3). The number of columns in the subplot grid.
    - fig_width: int, optional (default=20). The width of the entire plot figure (not the subplot width)
    - subplot_height: int, optional (default=4). The height of each subplot.
    - rotation: int, optional (default=45). The rotation of the x-axis labels.
    - strip: boolean, optional (default=False). Will strip single quotes from ends of column names
    - cat_cols: list, optional (default=None). A list of column names to treat as categorical variables. If not
        provided, inferred based on the unique count.
    - cont_cols: list, optional (default=None). A list of column names to treat as continuous variables. If not
        provided, inferred based on the unique count.
    - dtype_check: boolean, optional (default=True). If True, consider only numeric types (int64, float64) for
        continuous variables.
    - sample_size: float or int, optional (default=None). If provided and less than 1, the fraction of the data to
        sample. If greater than or equal to 1, the number of samples to draw.

    Returns: None
    """

    # Helper function to plot continuous variables
    def plot_continuous(df, cols, ncols, fig_width, subplot_height, strip, sample_size):
        nrows = ceil(len(cols) / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, nrows * subplot_height), constrained_layout=True)
        axs = np.array(axs).ravel()  # Ensure axs is always a 1D numpy array

        # Loop through all continuous columns
        for i, col in enumerate(cols):
            if sample_size:
                sample_count = int(len(df[col].dropna()) * sample_size)  # Calculate number of samples
                data = df[col].dropna().sample(sample_count)
            else:
                data = df[col].dropna()

            if strip:
                sns.stripplot(x=data, ax=axs[i])
            else:
                sns.histplot(data, ax=axs[i], kde=False)

            axs[i].set_title(f'{col}', fontsize=20)
            axs[i].tick_params(axis='x', rotation=rotation)
            axs[i].set_xlabel('')

        # Remove empty subplots
        for empty_subplot in axs[len(cols):]:
            empty_subplot.remove()

    # Helper function to plot categorical variables
    def plot_categorical(df, cols, ncols, fig_width, subplot_height, rotation, sample_size):
        nrows = ceil(len(cols) / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, nrows * subplot_height), constrained_layout=True)
        axs = np.array(axs).ravel()  # Ensure axs is always a 1D numpy array

        # Loop through all categorical columns
        for i, col in enumerate(cols):
            uv = df[col].value_counts().reset_index().rename(columns={col: 'number', 'index': 'name'})
            uv['perc'] = uv['number'] / uv['number'].sum()

            if sample_size:
                uv = uv.sample(sample_size)

            sns.barplot(data=uv, x='name', y='number', order=uv.sort_values('number', ascending=False).name, ax=axs[i])

            axs[i].set_title(f'{col}', fontsize=20)
            axs[i].tick_params(axis='x', rotation=rotation)
            axs[i].set_ylabel('Count')
            axs[i].set_xlabel('')

        # Remove empty subplots
        for empty_subplot in axs[len(cols):]:
            empty_subplot.remove()

    # Compute unique counts and identify categorical and continuous variables
    unique_count = df.nunique()
    if cat_cols is None:
        cat_cols = unique_count[unique_count <= n].index.tolist()
    if cont_cols is None:
        cont_cols = unique_count[unique_count > n].index.tolist()

    if dtype_check:
        cont_cols = [col for col in cont_cols if df[col].dtype in ['int64', 'float64']]

    if plot_type == 'cat' or plot_type == 'both':
        plot_categorical(df, cat_cols, ncols, fig_width, subplot_height, rotation, sample_size)
    if plot_type == 'cont' or plot_type == 'both':
        plot_continuous(df, cont_cols, ncols, fig_width, subplot_height, strip, sample_size)


def plot_charts_with_hue(df, plot_type='both', n=10, ncols=3, fig_width=20, subplot_height=4, rotation=0,
                         cat_cols=None, cont_cols=None, dtype_check=True, sample_size=None, hue=None, color_discrete_map=None, normalize=False, kde=False, multiple='layer'):
    """
    Version 0.1
    Plot barplots for categorical columns, or histograms for continuous columns, in a grid of subplots.
    Option to pass a 'hue' parameter to dimenions the plots by a variable/column of the dataframe.

    Parameters:
    - df: dataframe that contains the variables you want to analyze
    - plot_type: string, optional (default='both'). Type of charts to plot: 'cat' for categorical, 'cont' for continuous, 'both' for both
    - n: int (default=20). Threshold of unique values for categorical (equal or below) vs. continuous (above)
    - ncols: int, optional (default=3). The number of columns in the subplot grid.
    - fig_width: int, optional (default=20). The width of the entire plot figure (not the subplot width)
    - subplot_height: int, optional (default=4). The height of each subplot.
    - rotation: int, optional (default=45). The rotation of the x-axis labels.
    - cat_cols: list, optional (default=None). A list of column names to treat as categorical variables. If not provided, inferred based on the unique count.
    - cont_cols: list, optional (default=None). A list of column names to treat as continuous variables. If not provided, inferred based on the unique count.
    - dtype_check: boolean, optional (default=True). If True, consider only numeric types (int64, float64) for continuous variables.
    - sample_size: float or int, optional (default=None). If provided and less than 1, the fraction of the data to sample. If greater than or equal to 1, the number of samples to draw.
    - hue: string, optional (default=None). Name of the column to dimension by passing as 'hue' to the Seaborn charts.
    - color_discrete_map: name of array or array, optional (default=None). Pass a color mapping for the values in the 'hue' variable.
    - normalize: boolean, optional (default=False). Set to True to normalize categorical plots and see proportions instead of counts
    - kde: boolean, optional (default=False). Set to show KDE line on continuous countplots
    - multiple: 'layer', 'dodge', 'stack', 'fill', optional (default='layer'). Choose how to handle hue variable when plotted on countplots
    Returns: None
    """
    def plot_categorical(df, cols, ncols, fig_width, subplot_height, rotation, sample_size, hue, color_discrete_map, normalize):
        if sample_size:
            df = df.sample(sample_size)
        nplots = len(cols)
        nrows = nplots//ncols
        if nplots % ncols:
            nrows += 1

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, nrows*subplot_height), constrained_layout=True)
        axs = axs.ravel()

        for i, col in enumerate(cols):
            if normalize:
                # Normalize the counts
                df_copy = df.copy()
                data = df_copy.groupby(col)[hue].value_counts(normalize=True).rename('proportion').reset_index()
                sns.barplot(data=data, x=col, y='proportion', hue=hue, palette=color_discrete_map, ax=axs[i])
                axs[i].set_ylabel('Proportion', fontsize=12)
            else:
                sns.countplot(data=df, x=col, hue=hue, palette=color_discrete_map, ax=axs[i])
                axs[i].set_ylabel('Count', fontsize=12)
            axs[i].set_xlabel(' ', fontsize=12)
            axs[i].set_title(col, fontsize=16, pad=10)
            axs[i].tick_params(axis='x', rotation=rotation)

        # Remove empty subplots
        for empty_subplot in axs[nplots:]:
            fig.delaxes(empty_subplot)

    def plot_continuous(df, cols, ncols=3, fig_width=15, subplot_height=5, sample_size=None, hue=None, color_discrete_map=None, kde=False, multiple=multiple):
        if sample_size is not None:
            df = df.sample(sample_size)
        n = len(cols)
        nrows = int(np.ceil(n / ncols))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, nrows * subplot_height), constrained_layout=True)
        axs = axs.ravel()
        for i, col in enumerate(cols):
            if hue is not None:
                sns.histplot(data=df, x=col, hue=hue, palette=color_discrete_map, ax=axs[i], kde=kde, multiple=multiple)
            else:
                sns.histplot(data=df, x=col, ax=axs[i])
            axs[i].set_title(col, fontsize=16, pad=10)
            axs[i].set_ylabel('Count', fontsize=12)
            axs[i].set_xlabel(' ', fontsize=12)

    unique_count = df.nunique()
    if cat_cols is None:
        cat_cols = unique_count[unique_count <= n].index.tolist()
        if hue in cat_cols:
            cat_cols.remove(hue)
    if cont_cols is None:
        cont_cols = unique_count[unique_count > n].index.tolist()

    if dtype_check:
        cont_cols = [col for col in cont_cols if df[col].dtype in ['int64', 'float64']]

    if plot_type == 'cat' or plot_type == 'both':
        plot_categorical(df, cat_cols, ncols, fig_width, subplot_height, rotation, sample_size, hue, color_discrete_map, normalize)
    if plot_type == 'cont' or plot_type == 'both':
        plot_continuous(df, cont_cols, ncols, fig_width, subplot_height, sample_size, hue, color_discrete_map, kde, multiple)


def plot_corr(df, column, n, meth='pearson', size=(15, 8), rot=45, pal='RdYlGn', rnd=2):
    """
    Version 0.2
    Create a barplot that shows correlation values for one variable against others.
    Essentially one slice of a heatmap, but the bars show the height of the correlation
    in addition to the color. It will only look at numeric variables.

    Parameters:
    - df: dataframe that contains the variables you want to analyze
    - column: string. Column name that you want to evaluate the correlations against
    - n: int. The number of correlations to show (split evenly between positive and negative correlations)
    - meth: optional (default='pearson'). See df.corr() method options
    - size: tuple of ints, optional (default=(15, 8)). The size of the plot
    - rot: int, optional (default=45). The rotation of the x-axis labels
    - pal: string, optional (default='RdYlGn'). The color map to use
    - rnd: int, optional (default=2). Number of decimal places to round to

    Returns: None
    """
    # Calculate correlations
    corr = round(df.corr(method=meth, numeric_only=True)[column].sort_values(), rnd)

    # Drop column from correlations (correlating with itself)
    corr = corr.drop(column)

    # Get the most negative and most positive correlations, sorted by absolute value
    most_negative = corr.sort_values().head(n // 2)
    most_positive = corr.sort_values().tail(n // 2)

    # Concatenate these two series and sort the final series by correlation value
    corr = pd.concat([most_negative, most_positive]).sort_values()

    # Generate colors based on correlation values using a colormap
    cmap = plt.get_cmap(pal)
    colors = cmap((corr.values + 1) / 2)

    # Plot the chart
    plt.figure(figsize=size)
    plt.axhline(y=0, color='lightgrey', alpha=0.8, linestyle='-')
    bars = plt.bar(corr.index, corr.values, color=colors)

    # Add value labels to the end of each bar
    for bar in bars:
        yval = bar.get_height()
        if yval < 0:
            plt.text(bar.get_x() + bar.get_width() / 3.0, yval - 0.05, yval, va='top')
        else:
            plt.text(bar.get_x() + bar.get_width() / 3.0, yval + 0.05, yval, va='bottom')

    plt.title('Correlation with ' + column, fontsize=20)
    plt.ylabel('Correlation', fontsize=14)
    plt.xlabel('Other Variables', fontsize=14)
    plt.xticks(rotation=rot)
    plt.ylim(-1, 1)
    plt.show()


def split_dataframe(df, n):
    """
    Split a DataFrame into two based on the number of unique values in each column.

    Parameters:
    - df: DataFrame. The DataFrame to split.
    - n: int. The maximum number of unique values for a column to be considered categorical.

    Returns:
    - df_cat: DataFrame. Contains the columns of df with n or fewer unique values.
    - df_num: DataFrame. Contains the columns of df with more than n unique values.
    """
    df_cat = pd.DataFrame()
    df_num = pd.DataFrame()

    for col in df.columns:
        if df[col].nunique() <= n:
            df_cat[col] = df[col]
        else:
            df_num[col] = df[col]

    return df_cat, df_num


def thousands(x, pos):
    """
    Format a number with thousands separators.

    Parameters:
    - x: float. The number to format.
    - pos: int. The position of the number.

    Returns:
    - s: string. The formatted number.
    """
    s = '{:0,d}'.format(int(x))
    return s
