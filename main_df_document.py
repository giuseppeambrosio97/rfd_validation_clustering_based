import pandas as pd

from clustering_based_rfd_validation.lib.algorithms import k_clustering_based_RFD_validation, \
    rfd_brutal_force_validation
from clustering_based_rfd_validation.lib.data_structure import Rfd
from clustering_based_rfd_validation.lib.utils import dist_int_abs

if __name__ == '__main__':
    df = pd.read_csv('datasets/dataset_documento.CSV', delimiter=';')

    print(df)

    rfd = Rfd(['X'], 'Y',
              [3], 2,
              [dist_int_abs], dist_int_abs)
    rfds = [
        rfd,
    ]

    rfds_pred = k_clustering_based_RFD_validation(df, rfds, 4)

    rfds_real = []

    for rfd in rfds:
        rfds_real.append(rfd_brutal_force_validation(df,
                                                     rfd.b_LHS + [rfd.b_RHS],
                                                     rfd.LHS + [rfd.RHS],
                                                     rfd.f_LHS + [rfd.f_RHS]))

    print("---------------------------------------------------------------------------")
    print('OUTPUT:')
    for i in range(len(rfds)):
        print('RFD: ', i + 1, 'di ', len(rfds), ' \n', rfds[i])
        print('Output predetto: ', rfds_pred[i])
        print('Output reale: ', rfds_real[i])
        print("---------------------------------------------------------------------------")

    print("---------------------------------------------------------------------------")


