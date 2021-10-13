from operator import attrgetter

import numpy as np

from clustering_based_rfd_validation.lib.utils import dist_int_abs


class Rfd:
    """
    """

    def __init__(self, LHS, RHS, b_LHS, b_RHS, f_LHS, f_RHS):
        self.LHS = LHS
        self.RHS = RHS
        self.b_LHS = b_LHS
        self.b_RHS = b_RHS
        self.f_LHS = f_LHS
        self.f_RHS = f_RHS

    def __str__(self):
        return (
                'LHS ' + str(self.LHS) +
                ', RHS ' + str(self.RHS) +
                ', b_LHS ' + str(self.b_LHS) +
                ', b_RHS ' + str(self.b_RHS)
        )


class Matrix:
    """
    Struttura dati per memorizzare le sommarizzazioni delle tuple

    Variabili d'istanza:
      matrix (numpy.matrix)
      m (int): righe della matrice
      n (int): colonne della matrice
      soglie_LHS (float[])
      soglie_RHS (float)
      cluster_RHS (Wrapper_cluster_RHS[])
      cluster_LHS (Wrapper_cluster_LHS[])
      tuples: tuple da sommarizzare dell'ultimo datasets
    """

    def __init__(self, soglie):
        """
        Parameters:
          soglie (float[])
        """
        self.is_build = False
        self.soglie_LHS = soglie[:-1]
        self.soglie_RHS = soglie[-1]

    def get_intersect_cluster(self, i_RHS, j_LHS):
        return self.matrix[i_RHS, j_LHS]

    def build_matrix(self, cluster_RHS, cluster_LHS):
        """
        Parameters:
          cluster_RHS (Wrapper_cluster_RHS[])
          cluster_LHS (Wrapper_cluster_LHS[])
        """
        self.is_build = True
        self.cluster_RHS = cluster_RHS
        self.cluster_LHS = cluster_LHS
        cols = len(cluster_LHS)
        rows = len(cluster_RHS)
        # flag[i] = True se e solo se la colonna i ha un valore maggiore di 0
        self.flag = [False for i in range(cols)]
        self.matrix = np.zeros(dtype=int, shape=(rows, cols))

        return self.add_tuples()

    def add_tuples(self):
        """
        Return:
          (bool): True se e solo se l'aggiunta delle tupla t non viola la RFD
                  secondo la logica dell'algoritmo, False altrimenti
        """
        for id_tupla in self.tuples:
            id_LHS = None
            id_RHS = None

            for id, cluster in enumerate(self.cluster_LHS):
                if cluster.is_in_cluster(id_tupla):
                    id_LHS = id

            for id, cluster in enumerate(self.cluster_RHS):
                if cluster.is_in_cluster(id_tupla):
                    id_RHS = id

            if self.flag[id_LHS]:
                if self.matrix[id_RHS, id_LHS] != 0:
                    self.matrix[id_RHS, id_LHS] += 1
                else:
                    return False
            else:
                self.flag[id_LHS] = True
                self.matrix[id_RHS, id_LHS] += 1

        return True

    def merge_matrix(self, cluster_RHS, cluster_LHS):
        """
        Parameters:
          cluster_RHS (pandas.dataframe[]): cluster con tuple del datasets
          cluster_LHS (pandas.dataframe[]): cluster con tuple del datasets
        """

        # lista tuple da sommarizzare
        self.tuples = []

        for df in cluster_LHS:
            self.tuples = self.tuples + df.index.values.tolist()

        # wrapper
        for i in range(len(cluster_RHS)):
            cluster_RHS[i] = WrapperClusterRHS(cluster_RHS[i], self.soglie_RHS)

        for i in range(len(cluster_LHS)):
            cluster_LHS[i] = WrapperClusterLHS(cluster_LHS[i], self.soglie_LHS)

        # ordinamento
        cluster_RHS = sorted(cluster_RHS, key=attrgetter('cmin'))
        cluster_LHS = sorted(cluster_LHS, key=lambda cluster_LHS: cluster_LHS.cmin[0])

        # merge
        if not self.is_build:
            return self.build_matrix(cluster_RHS, cluster_LHS)

        self.cluster_RHS = self.merge_RHS(cluster_RHS)
        self.cluster_LHS = self.merge_LHS(cluster_LHS)

        # aggiunta tuple
        return self.add_tuples()

    def add_row(self, index):
        self.matrix = np.insert(self.matrix, index, np.zeros(self.matrix.shape[1]), axis=0)

    def add_col(self, index):
        self.matrix = np.insert(self.matrix, index, np.zeros(self.matrix.shape[0]), axis=1)
        self.flag.insert(index, False)

    def sum_cols(self, i, j):
        self.matrix[:, i] += self.matrix[:, j]
        self.matrix = np.delete(self.matrix, j, axis=1)
        del self.flag[j]

    def merge_RHS(self, cluster_RHS):
        """
        Complessità O(len(M_RHS)+len(m_RHS))

        Parameters:
          cluster_RHS (Wrapper_cluster_RHS[])

        Return:
          cluster_RHS (Wrapper_cluster_RHS[]): rispettando la proprietà P2
        """
        merged = []
        i = 0
        j = 0
        k = 0

        while i < len(self.cluster_RHS):
            while j < len(cluster_RHS):

                if self.cluster_RHS[i].merge(cluster_RHS[j]):  # merge
                    j += 1

                else:  # non merge
                    if self.cluster_RHS[i].cmin > cluster_RHS[j].cmin:
                        merged.append(cluster_RHS[j])
                        self.add_row(k)
                        j += 1
                        k += 1

                    else:
                        break

            merged.append(self.cluster_RHS[i])
            i += 1
            k += 1

        while j < len(cluster_RHS):
            merged.append(cluster_RHS[j])
            self.add_row(k)
            j += 1
            k += 1

        return merged

    def merge_LHS(self, cluster_LHS):
        """
        Parameters:
          cluster_LHS (Wrapper_cluster_LHS[])

        Return:
          (Wrapper_cluster_LHS[])
        """

        merged = self.__merge_sorted_lists(cluster_LHS)

        for i, aging_wrapper_LHS in enumerate(merged):
            if not aging_wrapper_LHS.flag:
                self.add_col(i)

        i = 0

        while (i + 1) < len(merged):
            if (merged[i].wc).merge(merged[i + 1].wc):
                self.sum_cols(i, i + 1)
                merged.pop(i + 1)
            else:
                i += 1

        for i, aging_wrapper_LHS in enumerate(merged):
            merged[i] = aging_wrapper_LHS.wc

        return merged

    class aging_wrapper_LHS:
        """
          Variabili di istanza:
            wc (WrapperClusterLHS)
            flag (bool) : True se wc è un WrapperClusterLHS già
                          presente nella matrice False altrimenti
        """

        def __init__(self, wc, flag):
            self.wc = wc
            self.flag = flag

    def __merge_sorted_lists(self, cluster_LHS):
        """
        Merge sort two sorted lists

        Parametrs:
          cluster_LHS (Wrapper_cluster_LHS[])

        Return:
          (__aging_wrapper_LHS[]): lista ordinata
        """
        sorted_list = []
        l1 = []
        for wrapper_cluster_LHS in self.cluster_LHS:
            l1.append(Matrix.aging_wrapper_LHS(wrapper_cluster_LHS, True))

        l2 = []
        for wrapper_cluster_LHS in cluster_LHS:
            l2.append(Matrix.aging_wrapper_LHS(wrapper_cluster_LHS, False))

        while l1 and l2:
            if l1[0].wc.cmin[0] <= l2[0].wc.cmin[0]:  # Compare both heads
                item = l1.pop(0)  # Pop from the head
                sorted_list.append(item)
            else:
                item = l2.pop(0)
                sorted_list.append(item)

        sorted_list.extend(l1 if l1 else l2)

        return sorted_list

    def print_matrix(self):
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                print(self.matrix[i][j], end='\t')
            print('|', self.cluster_RHS[i].count, end='\t')
            print()
        for j in range(self.matrix.shape[1]):
            print('----', end='\t')
        print()
        for j in range(len(self.cluster_LHS)):
            print(self.cluster_LHS[j].count, end='\t')
        print()


class WrapperClusterLHS:
    """
    Wrapper che contiene informazioni e metodi utili per un cluster su LHS

    Variabili d'istanza:
      items (set): cluster con gli indici delle tuple datasets
      beta (float): soglie
      cmin (float): valore minimo in items
      cmax (float): valore massimo in items
      count (int): numero di item
    """

    def __init__(self, items, beta):
        """
        Parameters:
          items (pandas.dataframe): cluster con gli indici delle tuple datasets
          beta (float[]): soglie
        """
        self.cmin = items.min().values.astype(np.float)
        self.cmax = items.max().values.astype(np.float)
        self.count = items.shape[0]
        self.items = items
        self.beta = beta

    def __str__(self):
        return ("[count = " + str(self.count) +
                ", min = " + str(self.cmin) +
                ", max = " + str(self.cmax) +
                ",\n items = \n" + str(self.items) +
                ", beta = " + str(self.beta) +
                "]")

    def merge(self, wrapper_cluster_LHS):
        """
        Parameters:
          wrapper_cluster_LHS (WrapperClusterLHS): da mergare con self

        Return:
          (bool): True se e solo self e wrapper_cluster_LHS possono essere uniti
                  soddisfando P1 FALSE altrimenti

        ###############
        [stima_alpha1,alpha2]: se consideremo non solo soddisfare P1
        """
        for i in range(len(self.beta)):
            if (wrapper_cluster_LHS.cmax[i] < self.cmin[i] - self.beta[i]
                    or
                    self.cmax[i] + self.beta[i] < wrapper_cluster_LHS.cmin[i]):
                return False

        for i in range(len(self.beta)):
            self.cmin[i] = min(wrapper_cluster_LHS.cmin[i], self.cmin[i])
            self.cmax[i] = max(wrapper_cluster_LHS.cmax[i], self.cmax[i])

        self.count += wrapper_cluster_LHS.count
        self.items = self.items.append(wrapper_cluster_LHS.items)

        return True

    def is_in_cluster(self, id_tupla):
        """
        Parameters:
          id_tupla (int): id della tupla

        Return:
          (bool): True se la tupla è nel cluster, False altrimenti
        """
        return id_tupla in self.items.index.values.tolist()


class WrapperClusterRHS:
    """
    Wrapper che contiene informazioni e metodi utili per un cluster su RHS

    items (set): cluster con gli indici delle tuple datasets
    beta (float[]): soglia della RFD sul RHS
    count (int): numero item
    centro (float): centro dell'intervallo
    cmin (float): valore minimo in items
    cmax (float): valore massimo in items
    e_sup (float) : estremo superiore massimo
    e_inf (float): estremo inferiore minimo
    """

    def __init__(self, items, beta):
        """
        Parameters:
          items (pandas.datafram): cluster con gli indici delle tuple datasets
          beta (float[]): soglia della RFD sul RHS
        """
        self.cmin = float(items.min())
        self.cmax = float(items.max())

        self.count = items.shape[0]
        self.items = items
        self.beta = beta

    def __str__(self):
        return ("[count = " + str(self.count) +
                ", min = " + str(self.cmin) +
                ", max = " + str(self.cmax) +
                ",\n items = \n" + str(self.items) +
                ", beta = " + str(self.beta) +
                "]")

    def merge(self, wrapper_cluster_RHS):
        """
        Parameters:
          wrapper_cluster_RHS (WrapperClusterRHS)

        Return:
          (bool): True il merge soddisfa P2, False altrimenti

        ###############
        [stima_alpha1,alpha2]:se consideremo non solo soddisfare P2 ma anche max P1
        """
        if (dist_int_abs(min(wrapper_cluster_RHS.cmin, self.cmin),max(wrapper_cluster_RHS.cmax, self.cmax)) < self.beta):
            self.count += wrapper_cluster_RHS.count
            self.cmin = min(wrapper_cluster_RHS.cmin, self.cmin)
            self.cmax = max(wrapper_cluster_RHS.cmax, self.cmax)
            self.items = self.items.append(wrapper_cluster_RHS.items)
            return True
        else:
            return False

    def is_in_cluster(self, id_tupla):
        """
        Parameters:
          id_tupla (int): id della tupla

        Return:
          (bool): True se la tupla è nel cluster, False altrimenti
        """
        return id_tupla in self.items.index.values.tolist()
