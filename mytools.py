import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
  
def get_unique(df, n=20, sort='none', list=True, strip=False, count=False, percent=False, plot=False, cont=False):
    """
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
            uv = pd.DataFrame({'orig':orig, 'name':name, 'number':number, 'perc':perc})

            # Sort the unique values by name or count, if specified
            if sort == 'name':
                uv = uv.sort_values(by='name', ascending=True)
            elif sort == 'count':
                uv = uv.sort_values(by='number', ascending=False)
            elif sort == 'percent':
                uv = uv.sort_values(by='perc', ascending=False)

            # Print out the list of unique values for each variable
            if list == True:
                print(f"\n{var_name} has {unique_count} unique values:\n")
                for w, x, y, z in uv.itertuples(index=False):
                    # Decide on to use stripped name or not
                    if strip == True:
                        w = x
                    # Put some spacing after the value names for readability
                    w_str = str(w)
                    w_pad_size = uv.name.str.len().max() + 7
                    w_pad = w_str + " "*(w_pad_size - len(w_str))
                    y_str = str(y)
                    y_pad_max = uv.number.max()
                    y_pad_max_str = str(y_pad_max)
                    y_pad_size = len(y_pad_max_str) + 3
                    y_pad = y_str + " "*(y_pad_size - len(y_str))
                    if count and percent:
                         print("\t" + str(w_pad) + str(y_pad) + str(z) + "%")
                    elif count:
                        print("\t" + str(w_pad) + str(y))
                    elif percent:
                        print("\t" + str(w_pad) + str(z) + "%")
                    else:
                        print("\t" + str(w))

            # Plot countplot if plot=True
            if plot == True:
                print("\n")
                if strip == True:
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
            if plot == True:
                print("\n")
                sns.histplot(data=df, x=var_name)
                #plt.title(var_name)
                #plt.xlabel('')
                #plt.ylabel('')
                #plt.xticks(rotation=45)
                plt.show()
                
def plot_charts(df, n=10, ncols=3, figsize=(20, 40), rotation=45):
    """
    Plot histograms for each column in a DataFrame in a grid of subplots.

    Parameters:
    - df: dataframe that contains the variables you want to analyze
    - n: int (default=20). Threshold of unique values for categorical (equal or below) vs. continuous (above)
    - ncols: int, optional (default=3). The number of columns in the subplot grid.
    - figsize: tuple of ints, optional (default=(20, 20)). The size of the entire plot figure.
    - rotation: int, optional (default=45). The rotation of the x-axis labels.

    Returns: None
    """
    # Define number of rows and columns for subplot grid
    num = len(df.columns)
    nrows = num // ncols if num % ncols == 0 else num // ncols + 1

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    var_list = df.nunique(axis=0)
            
    for i in range(len(var_list)):
    #for i, col in enumerate(df.columns):
        r, c = i // ncols, i % ncols
        var_name = var_list.index[i]
        unique_count = var_list[i]
        
        if unique_count <= n:
            number = df[var_name].value_counts(dropna=False)
            perc = round(number / df.shape[0] * 100, 2)
            # Copy the index to a column
            orig = number.index
            # Strip out the single quotes
            name = [str(n) for n in number.index]
            name = [n.strip('\'') for n in name] 
            # Store everything in dataframe uv for consistent access and sorting
            uv = pd.DataFrame({'orig':orig, 'name':name, 'number':number, 'perc':perc})
            # Draw a barplot
            sns.barplot(data=uv, x='name', y='number', order=uv.sort_values('number', ascending=False).name, ax=axs[r, c])
            axs[r, c].set_title(var_name, fontdict={'fontsize':16})
            axs[r, c].set_xlabel('')
            for label in axs[r, c].get_xticklabels():
                label.set_rotation(rotation)
        else:
            sns.histplot(data=df, x=var_name, ax=axs[r, c])
            axs[r, c].set_title(var_name, fontdict={'fontsize':16})

    for j in range(num, nrows*ncols):
        fig.delaxes(axs.flatten()[j])   
    plt.tight_layout()
    plt.show()

def plot_corr(df, column, meth='pearson', size=(15, 8), rot=45, pal='RdYlGn', rnd=2):
    """
    Create a barplot that shows correlation values for one variable against others.
    Essentially one slice of a heatmap, but the bars show the height of the correlation
    in addition to the color. It will only look at numeric variables.

    Parameters:
    - df: dataframe that contains the variables you want to analyze
    - column: string. Column name that you want to evaluate the correlations against
    - meth: optional (default='pearson'). See df.corr() method options
    - size: tuple of ints, optional (default=(15, 8)). The size of the plot
    - rot: int, optional (default=45). The rotation of the x-axis labels
    - pal: string, optional (default='RdYlGn'). The color map to use
    - rnd: int, optional (default=2). Number of decimel places to round to

    Returns: None
    """
    # Calculate correlations
    corr = round(df.corr(method=meth, numeric_only=True)[column].sort_values(), rnd)

    # Drop column from correlations (correlating with itself)
    corr = corr.drop(column)

    # Generate colors based on correlation values using a colormap
    cmap = plt.get_cmap(pal)
    colors = cmap((corr.values + 1) / 2)

    # Plot the chart
    plt.figure(figsize=size)
    plt.axhline(y = 0, color = 'lightgrey', alpha=0.8, linestyle = '-')
    bars = plt.bar(corr.index, corr.values, color=colors)

    # Add value labels to the end of each bar
    for bar in bars:
        yval = bar.get_height()
        if yval < 0:
            plt.text(bar.get_x() + bar.get_width()/3.0, yval - 0.05, yval, va='top') 
        else:
            plt.text(bar.get_x() + bar.get_width()/3.0, yval + 0.05, yval, va='bottom')

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
