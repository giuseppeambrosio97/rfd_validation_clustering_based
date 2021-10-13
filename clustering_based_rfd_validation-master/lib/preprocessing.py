import networkx as nx


def get_similarity_graph(ds, b, attrs, f):
    """
    Parameters:
      ds (Pandas.dataframe): datasets i-esimo
      b (float[]): lista ordinata soglie di similarità
      attrs (str[]): lista ordinata attributi da confrontare
      f (func[]): lista ordinata funzioni di distanza

    Return:
      (nx.Graph)
    """
    G = nx.Graph()

    for i, tupla in ds.iterrows():
        G.add_node(i, value=tupla[attrs].to_list())
        for j in range(i + 1, ds.index.stop):
            flag = True
            for k, attr in enumerate(attrs):
                if f[k](ds[attr][i], ds[attr][j]) > b[k]:
                    flag = False
                    break
            if flag:
                G.add_edge(i, j)
    return G
