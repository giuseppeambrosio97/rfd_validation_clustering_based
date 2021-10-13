import random

import networkx as nx
from networkx.algorithms.approximation import clique as nx_aprox

from lib.data_structure import Rfd, Matrix
from lib.preprocessing import get_similarity_graph
from lib.utils import dist_int_abs, stream_generator


def p1_max_p2(G):
    """
    Parameters:
      G (nx.Graph): grafo delle distanze su LHS

    Return:
      (set[]): cluster con gli indici delle tuple datasets
    """

    lista_componenti = [c for c in nx.connected_components(G)]

    return lista_componenti


def p2_max_p1(G):
    """
    Parameters:
      G (nx.Graph): grafo delle distanze su RHS

    Return:
      (set[]): cluster con gli indici delle tuple datasets
    """
    C = []

    F = nx.Graph()
    F.add_nodes_from(G)
    F.add_edges_from(G.edges())

    while len(F.nodes()) > 0:
        new_max_clique = nx_aprox.max_clique(F)
        F.remove_nodes_from(new_max_clique)
        C.append(new_max_clique)

    return C


def rfd_validation_offline(ds, b, attrs, f):
    """
    Parameters:
      ds (Pandas.dataframe): datasets i-esimo
      b (float[]): lista ordinata soglie di similarità
      attrs (str[]): lista ordinata attributi da confrontare
      f (func[]): lista ordinata funzioni di distanza

    Return:
      flag (bool): True RFD valida, False altrimenti
      cluster_RHS (set[]): cluster con gli indici delle tuple datasets
      cluster_LHS (set[]): cluster con gli indici delle tuple datasets
    """

    flag = rfd_brutal_force_validation(ds, b, attrs, f)
    df_cluster_lhs = None
    df_cluster_rhs = None

    if flag:
        # costruzione grafi
        G_LHS = get_similarity_graph(ds, b[:-1], attrs[:-1], f[:-1])
        G_RHS = get_similarity_graph(ds, b[-1:], attrs[-1:], f[-1:])

        # clustering
        cluster_LHS = p1_max_p2(G_LHS)
        cluster_RHS = p2_max_p1(G_RHS)

        # estrazione datasets
        df_cluster_rhs = [ds.loc[list(x)][attrs[-1]] for x in cluster_RHS]
        df_cluster_lhs = [ds.loc[list(x)][attrs[:-1]] for x in cluster_LHS]

    return (
        flag,
        df_cluster_rhs, df_cluster_lhs
    )


def rfd_validate_online(ds, b, attrs, f, matrix):
    """
    Parameters:
      ds (Pandas.dataframe): datasets i-esimo
      b (float[]): lista ordinata soglie di similarità
      attrs (str[]): lista ordinata attributi da confrontare
      f (func[]): lista ordinata funzioni di distanza
      matrix (Matrix): matrice che sommarizza i precedenti datasets

    Return:
      (bool): True se la RFD è valida su i datasets finora considerati,
              False altrimenti
    """
    flag, cluster_RHS, cluster_LHS = rfd_validation_offline(ds, b, attrs, f)

    if flag:
        result = matrix.merge_matrix(cluster_RHS, cluster_LHS)
        return result
    else:
        return False


def k_clustering_based_RFD_validation(dataset, rfds, split):
    """
    Parameters:
      dataset (Pandas.dataframe): intero datasets
      rfds (Rfd[]): lista di RFD da validare
      split (int)

    Return:
      rfds_to_validate (bool[]): rfds_to_validate[i]=True se la RFD vale, False altrimenti
    """
    matrixs = []
    rfds_to_validate = [True for _ in range(len(rfds))]

    soglie_rfds = []
    funzioni_rfds = []
    attributi_rfds = []

    for rfd in rfds:
        soglie_rfd = rfd.b_LHS + [rfd.b_RHS]
        soglie_rfds.append(soglie_rfd)
        attributi_rfds.append(rfd.LHS + [rfd.RHS])
        funzioni_rfds.append(rfd.f_LHS + [rfd.f_RHS])
        matrixs.append(Matrix(soglie_rfd))

    for d_i in stream_generator(dataset, split):

        for i, rfd_i_is_to_validate in enumerate(rfds_to_validate):
            if rfd_i_is_to_validate:

                result = rfd_validate_online(d_i,
                                             soglie_rfds[i],
                                             attributi_rfds[i],
                                             funzioni_rfds[i],
                                             matrixs[i])

                if not result:
                    rfds_to_validate[i] = False

    return rfds_to_validate


def rfd_brutal_force_validation(ds, b, attrs, f):
    """
    Parameters:
      ds (Pandas.dataframe): datasets i-esimo
      b (float[]): lista ordinata soglie di similarità
      attrs (str[]): lista ordinata attributi da confrontare
      f (func[]): lista ordinata funzioni di distanza

    Return:
    --------
      (bool): True se la RFD è valida, False altrimenti
      (str): Commento sulla RFD che non vale
    """
    for i, _ in ds.iterrows():
        for j in range(i + 1, ds.index.stop):
            simili_LHS = True
            for k in range(len(b) - 1):
                # df[col][row]
                if f[k](ds[attrs[k]][i], ds[attrs[k]][j]) > b[k]:
                    simili_LHS = False
                    break
            if simili_LHS:
                if f[-1](ds[attrs[-1]][i], ds[attrs[-1]][j]) > b[-1]:
                    return (False,
                            "La RFD non vale, i =", i, " j =", j)
    return (True,
            "La RFD vale")


def rfd_generator(df, num_rfd_to_gen=1, rnd_seed=None):
    """
    """
    if rnd_seed:
        random.seed(rnd_seed)

    rfds = []

    for _ in range(num_rfd_to_gen):
        k = random.randint(2, len(df.columns))

        cols = random.sample(range(len(df.columns)), k)
        cols = [df.columns[i] for i in cols]

        lhs = cols[:-1]
        rhs = cols[-1:]

        beta = random.randint(2, 10)  ## GENERAZIONE SOGLIA
        beta = [beta for _ in range(k)]

        funcs = [dist_int_abs for _ in range(k)]

        rfds.append(Rfd(lhs, rhs,
                        beta[:-1], beta[-1:],
                        funcs[:-1], funcs[-1:]
                        )
                    )

    return rfds
