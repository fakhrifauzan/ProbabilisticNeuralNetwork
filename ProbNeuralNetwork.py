#======= PROBABILISTIC NEURAL NETWORK =========
# Machine Learning Task 1.3 (2018)
# Fakhri Fauzan
# IF - 39 - 10
# 1301154374
# Informatics Engineering, Telkom University
#==============================================

import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class ProNN :
    def __init__(self):
        self.training = []
        self.testing = []
        self.result = []
        self.n = [0, 0, 0]
        self.prob = [0.0, 0.0, 0.0]
        self.sigma = [0.0, 0.0, 0.0]

    def visualize(self, file):
        x = []
        y = []
        z = []
        count = 0

        with open(file, "r") as file:
            tLine = file.readlines()
            for t in tLine:
                if (count > 0):
                    row = t.split()
                    if (int(row[3]) == 0):
                        x.append([float(row[0]), float(row[1]), float(row[2])])
                    elif (int(row[3]) == 1):
                        y.append([float(row[0]), float(row[1]), float(row[2])])
                    else:
                        z.append([float(row[0]), float(row[1]), float(row[2])])
                count = count + 1
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)

        fig = plt.figure()
        data = fig.add_subplot(111, projection='3d')
        data.set_xlabel('Attr 1')
        data.set_ylabel('Attr 2')
        data.set_zlabel('Attr 3')
        data.scatter(x[:,0], x[:,1], x[:,2], c='r', marker='o')
        data.scatter(y[:,0], y[:,1], y[:,2], c='g', marker='+')
        data.scatter(z[:,0], z[:,1], z[:,2], c='b', marker='^')
        plt.show()

    def loadTraining(self, file):
        train = []
        count = 0
        with open(file, "r") as file:
            tLine = file.readlines()
            for x in tLine:
                if (count > 0):
                    row = x.split()
                    train.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
                    self.n[int(row[3]) - 1] = int(self.n[int(row[3]) - 1]) + 1
                count = count + 1
        self.training = sorted(train, key=lambda train:train[3], reverse=False)

    def loadTesting(self, file):
        count = 0
        with open(file, "r") as file:
            tLine = file.readlines()
            for x in tLine:
                if (count > 0):
                    row = x.split()
                    self.testing.append([float(row[0]), float(row[1]), float(row[2])])
                count = count + 1

    def writeResultTesting(self, file):
        file = open(file, "w")
        for i in range(len(self.result)):
            file.write(str(self.result[i]) + "\n")
        file.close()

    def smoothing(self, g=None):
        if g is None:
            g = 1.4
        dtot1 = []
        dtot2 = []
        dtot3 = []
        for i in range(len(self.training)):
            fmd = []
            for j in range(len(self.training)):
                a = 0
                if (self.training[i] != self.training[j] and self.training[i][3] == self.training[j][3]) :
                    for z in range(0, 3):
                        a = a + pow((float(self.training[j][z]) - float(self.training[i][z])), 2)
                    d = math.sqrt(a)
                    fmd.append(d)
            if (self.training[i][3] == 1) :
                dtot1.append(min(fmd))
            elif (self.training[i][3] == 2) :
                dtot2.append(min(fmd))
            else :
                dtot3.append(min(fmd))
        self.sigma = [float((g * sum(dtot1)) / len(dtot1)), float((g * sum(dtot2)) / len(dtot2)), float((g * sum(dtot3)) / len(dtot3))]

    def main(self) :
        #fungsi g(x)
        for z in range(len(self.testing)):
            self.prob = [0.0, 0.0, 0.0]
            for i in range(len(self.training)):
                a = 0.0
                p = 3
                kelas = int(self.training[i][3])
                for j in range(0,3):
                    a = a + float(pow((float(self.testing[z][j]) - float(self.training[i][j])),2))
                b = 2*(pow(self.sigma[kelas],2))
                dbp = pow(2*math.pi,p/2)
                dbs = pow(self.sigma[kelas],p)
                dbn = self.n[kelas]
                # print("DBA = " + str(dbp) + " | DBS = " + str(dbs) + " | DBN = " + str(dbn))
                d = 1 / (dbp * dbs * dbn)
                form = math.exp(-(a/b))
                x = d*form
                # print("A = " + str(a) + " | B = " + str(b) + " | Form = " + str(form) + " | X = " + str(x) + " | Kelas = " + str(kelas))
                self.prob[kelas] = self.prob[kelas] + x
            self.result.append(self.decision())

    def decision(self):
        index = self.prob.index(max(self.prob))
        return index

    def getAccuracy(self, file, g):
        train = []
        expected = []
        count = 0
        with open(file, "r") as file:
            tLine = file.readlines()
            for x in tLine:
                if (count > 0):
                    row = x.split()
                    if (count < 101) :
                        train.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
                        self.n[int(row[3]) - 1] = int(self.n[int(row[3]) - 1]) + 1
                    else :
                        expected.append(int(row[3]))
                        self.testing.append([float(row[0]), float(row[1]), float(row[2])])
                count = count + 1
            self.training = sorted(train, key=lambda train: train[3], reverse=False)
        self.smoothing(g)
        self.main()
        cSama = 0.0
        for x in range(len(self.result)):
            if (self.result[x] == expected[x]):
                cSama += 1
        akurasi = (cSama / len(expected)) * 100
        return akurasi

# .:: MAIN PROGRAM ::.
#======= Load Data =========
fileT = "data_train_PNN.txt"
fileTs = "data_test_PNN.txt"
#===========================

#======== PROSES OBSERVASI MENCARI NILAI G MAKSIMUM AKURASI ========
g = []
a = np.linspace(0.1, 2.0, num=20, endpoint=True)
fileOb = open("observasi.txt", "w")
for i in range(len(a)) :
    g.append(round(a[i],2))
# print ("Himpunan G : " + str(g))
fileOb.write("Himpunan G : " + str(g))
for k in range(len(g)) :
    pnn = ProNN()
    fileOb.write("G = " + str(g[k]) + " | Akurasi : " + str(pnn.getAccuracy(fileT, g[k])) + "%" + "\n")
    fileOb.write("Smoothing Value[0,1,2] : " + str(pnn.sigma) + "\n")
fileOb.close()
#====================================================================

# ========== PROSES KLASIFIKASI DATA TESTING ==========
pnn = ProNN()
pnn.loadTraining(fileT)
pnn.loadTesting(fileTs)
pnn.smoothing()
print("Smoothing Value : " + str(pnn.sigma))
pnn.main()
pnn.writeResultTesting("prediksi.txt")
print("Data Testing Berhasil di Klasifikasi!")
pnn.visualize(fileT)
# ======================================================