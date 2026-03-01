def tambah_satu(x):
    return x + 1

data = [1, 2, 3, 4, 5]

hasil = []
for angka in data:
    hasil.append(tambah_satu(angka))

print(hasil)