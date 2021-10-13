def dist_int_abs(a: int, b: int):
    return abs(a - b)


def stream_generator(dataset, chunk_size):
    for i in range(0, len(dataset), chunk_size):
        yield dataset[i:i + chunk_size]
