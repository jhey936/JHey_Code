import pandas as pd
import networkx as nx

data = {'Chemical Representation': ["C-c=c-c-c-", "c-c=o-", "C-O-C=N-"],
       'Frequency%':['8%', '11%', '60%'],
        'Total number': [5, 3, 8],
        'Semantics': ['54010', '32100', '43290'],
        'Graph Objects': ["0", "0", "0"]}


def make_data_frame():
    df = pd.DataFrame(data, columns=['Chemical Representation', 'Frequency%', 'Total number', 'Semantics', 'Graph Objects'])

    return df

# if __name__ == "__main__":
