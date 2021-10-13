# Clustering Based RFD Validation

Per installare le dipedenze usare il file  `requirements.txt`.
Nel `main` vi è un esempio di esecuzione che richiama la funzione `k_clustering_based_RFD_validation`.

```py
k_clustering_based_RFD_validation(dataset, parametri, rfd_dir_path)
  Parameters:
        dataset (Pandas.dataframe): dataset
        parametri ((int,int)[]): lista di coppie (slice,#rfd) su cui eseguire k_clustering_based_RFD_validation
        rfd_dir_path (string): path contenente i file delle RFD da validare nel formato dime (es. RFD_C0_E0_breast-cancer-wisconsin_pv.txt)
```