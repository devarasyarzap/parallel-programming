from multiprocessing import Pool

def tambah_satu(x):
    return x + 1

data = [1, 2, 3, 4, 5]

if __name__ == "__main__":
    with Pool() as p:
        hasil = p.map(tambah_satu, data)

    print(hasil)