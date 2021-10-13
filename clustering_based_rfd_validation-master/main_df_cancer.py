import time

import pandas as pd
from sklearn.metrics import confusion_matrix

from lib.algorithms import k_clustering_based_RFD_validation
from lib.io import get_k_rfd_from_dir


def k_clustering_based_RFD_validation_parametric(dataset, parametri, rfd_dir_path):
    """
      Parameters:
        dataset (Pandas.dataframe): datasets
        parametri ((int,int)[]): lista di coppie (slice,#rfd) su cui eseguire
                                k_clustering_based_RFD_validation
        rfd_dir_path (string): path contenente i file delle RFD da validare nel formato definito da DIME
    """

    for params in parametri:
        print("*****************************************************************************")
        slice_dataset = params[0]
        numero_rfd = params[1]
        print("ESECUZIONE CON slice = ", slice_dataset)
        print("ESECUZIONE CON numero rfd = ", numero_rfd)

        schema_relazione = dataset.columns.values
        rfds = get_k_rfd_from_dir(rfd_dir_path,
                                  schema_relazione, k=numero_rfd)

        start_time = time.time()

        rfds_pred = k_clustering_based_RFD_validation(dataset, rfds, slice_dataset)

        print("\ntempo finale ", time.time() - start_time)

        rfds_true = [True for _ in range(len(rfds))]

        tn, fp, fn, tp = confusion_matrix(rfds_true, rfds_pred, labels=[False, True]).ravel() / len(rfds)
        print('\ntn = ', tn)
        print('fp = ', fp)
        print('fn = ', fn)
        print('tp = ', tp)
        print("*****************************************************************************")


if __name__ == '__main__':
    rfd_dir_path = 'datasets/dime_rfd/breast-cancer-wisconsin_pv/'
    df_cancer = pd.read_csv('datasets/breast-cancer-wisconsin_pv.csv', delimiter=';')

    print(df_cancer)

    parameters = [
        (5, 5),
        (10, 60),
        (70, 56),
    ]

    k_clustering_based_RFD_validation_parametric(df_cancer, parameters, rfd_dir_path)
