from multiprocessing import Process

def tugas1():
    print("Tugas 1 jalan")

def tugas2():
    print("Tugas 2 jalan")

if __name__ == "__main__":
    p1 = Process(target=tugas1)
    p2 = Process(target=tugas2)

    p1.start()
    p2.start()

    p1.join()
    p2.join()