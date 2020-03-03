import pandas as pd
import numpy as np
import argparse
from scipy.stats import zscore
import seaborn as sns


# Feb. 25 2020 Yunh. Kang
# add --input

def parse_args():
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Glove HighTemp data receiver")
    ap.add_argument("-i", "--input", required=True,
                    help="input csv")
    return vars(ap.parse_args())


# z-score 4
def z_score(df):
    df.columns = [x + "_zscore" for x in df.columns.tolist()]
    return ((df - df.mean())/df.std(ddof=0))

def z_score_1(df):
    # z-score 1
    from scipy.stats import zscore
    df_zscore = df.apply(zscore)
    print(df_zscore.head())

def z_score_2(df):
    # z-score 2
    # now iterate over the remaining columns and create a new zscore column
    cols = list(df.columns)
    for col in cols:
        col_zscore = col + '_zscore'
        df[col_zscore] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    print(df.head())

def z_score_3(df):
    # z-score 3
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols].apply(zscore)
    print(df[numeric_cols].head())

def plot_zscore(df, df1):
    import matplotlib.pyplot as plt


    df[:]['error_zscore'].plot(kind='box', color='red', legend='error')
    plt.show()



    df1[:]['observation'].plot(kind='line', color='blue', legend='observation')
    df1[:]['reference'].plot(kind='line', color='green', legend='reference')
    plt.show()

    plt.hist(df1[:]['observation'], color='blue', edgecolor='black', bins=20)
    #df1[:]['observation'].plot(kind='kde', legend='observation kde')
    #plt.hist(df1[:]['observation'], color='blue', edgecolor='black', bins=10)

    plt.show()

    #ax = df1[:]['observation'].plot(kind='kde', color='blue', legend='observation kde')

    #ax = df1[:]['observation'].plot(kind='hist', legend='observation hist')
    #df1[:]['observation'].plot(kind='kde', legend='observation kde')

    sns.distplot(df1[:]['observation'], hist=True, kde=True,
                 bins=int(180 / 5), color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})
    plt.show()

def main(args):
    df = pd.read_csv(args["input"])
    df1 = df.copy()

    df1.drop(df[df['observation'] == 0.0].index, inplace=True)
    print ("z_score")
    print(z_score(df).head())
    print ("z_score\n")

    z_score_2(df)
    z_score_3(df)
    plot_zscore(df, df1)



if __name__ == '__main__':
    args = parse_args()
    main(args)