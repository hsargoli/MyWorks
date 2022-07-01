import matplotlib.image as mpim
import matplotlib.pyplot as plt
import glob
import numpy as np

# sorry for persian comments its for old era
# read pic from a file
def read(path):
    data = []
    for i in glob.glob(path, recursive=True):
        a = mpim.imread(i)
        a = a[:, :, 0]
        data.append(a)
    df = []
    # change    0 to 1    & 255 to -1 (255  white - 0 black)
    for j in range(len(data)):
        for k in data[j]:
            y = [
                i if i > 10 else 1 for i in k
            ]  # its persian comment yani age addi balatar az 10 bod nadide begiresh va age kamtar bod be 1 tagheeresh bede
            z = [i if i < 2 else -1 for i in y]
            df.append(z)
    df = np.array(df)
    df = np.reshape(
        df, (len(data), len(data[0]), len(data[0]))
    )  # tagheer shape har tasvir be (tedad tasavir, teedad addaad dar radif )
    return df


# tak boodi kardan tasavire yek arraye
def make1d(data):
    d1d = []
    for i in data:
        a = np.reshape(i, (25))
        d1d.append(a)
    d1d = np.array(d1d)
    return d1d


def list_predict(INdata, outdata, initw):
    bias = 1
    s = []
    for i in range(len(INdata)):
        wx = []
        for j in range(len(INdata[0])):
            a = INdata[i][j] * initw[j]
            wx.append(a)
        Sum = np.sum(wx) + bias
        s.append(Sum)
        del wx
    yhads = []
    for i in s:
        if i >= 0:
            yhads.append(1)
        elif i < 0:
            yhads.append(-1)
    return yhads


def upW(xdata, yhads, yorg, wazn, learnRate):
    ytotal = list(zip(yorg, yhads))
    newW = []
    for i in range(len(ytotal)):
        if ytotal[i][0] != ytotal[i][1]:
            print("bad index at: ", i, " ", ytotal[i][0], ytotal[i][1])
            for k in range(len(wazn)):
                for u in range(len(yhads)):
                    wj = wazn[k] + ((xdata[u][k]) * (yhads[u]) * (learnRate))
                newW.append(wj)
            print("newWeight", newW)
            break
        else:
            newW = wazn
    return newW


# ----


def score(yorg, yhads):
    a = list(zip(yhads, yt))
    score = 0
    for i, j in a:
        if i == j:
            score += 1
    print("(predicted Y  ,  orginal Y)\n", a)
    acc = score / len(yorg)
    print("Accuracy: ", acc)
    return acc


def tekrar(inData, outData, wazn, LR, num):
    print("Weights  :", wazn)
    hads0 = list_predict(inData, outData, wazn)
    newW = upW(inData, outData, hads0, wazn, LR)
    # newW=upW_adline(inData,outData,hads0,wazn,LR)  #adaline

    try:

        x = 0
        while x < num:

            hads = list_predict(inData, outData, newW)
            newW = upW(inData, outData, hads, newW, LR)
            # newW=upW_adline(inData,outData,hads,newW,LR)  #adaline

            # print('hads va asli \n\n',list(zip(hads,outData)))

            x += 1
    except:
        print("end")
    return newW


def read_one_img(path):
    a = mpim.imread(path)
    a = a[:, :, 0]
    a = np.array(a)
    b = []
    for j in a:
        y = [i if i > 10 else 1 for i in j]
        z = [i if i < 2 else -1 for i in y]
        b.append(z)
    b = np.reshape(b, (25))
    return b


def predict(x, w):
    a = []
    for i in range(len(w)):
        a = x[i] * w[i]
    Sum = np.sum(a)
    if Sum >= 0:
        yhads = 1
    elif Sum < 0:
        yhads = -1
    return yhads


def test(path, weight):
    q1 = read_one_img(path)
    H = predict(q1, weight)
    return H


# ostad natonestam javab khobi begiram an adaline
# faghat update wazn ro tagheer dadam vali hameye wazn ha sefr mishan!
def upW_adline(xdata, yhads, yorg, wazn, learnRate):
    ytotal = list(zip(yorg, yhads))
    newW = []
    for i in range(len(ytotal)):
        if ytotal[i][0] != ytotal[i][1]:
            print("bad index: ", i, " ", ytotal[i][0], (ytotal[i][1]))
            for k in range(len(wazn)):
                for u in range(len(yhads)):
                    wj = wazn[k] + ((xdata[u][k]) * (-yorg[u] + yhads[u]) * (learnRate))
                newW.append(wj)
            print("newWeight", newW)
            break
        else:
            newW = wazn
    return newW


xdf = read("dsox/X*.jpg")
odf = read("dsox/O*.jpg")
xdf1 = make1d(xdf)
odf1 = make1d(odf)


yxdata = np.ones(len(xdf))
yodata = np.ones(len(odf)) * -1

xtest = read("dsox/1/X*.jpg")
otest = read("dsox/1/O*.jpg")
xt = make1d(xtest)
ot = make1d(otest)
yxt = np.ones(len(xt))
yot = np.ones(len(ot)) * -1

# tajmii data ha
dst = np.concatenate((odf1, xdf1), axis=0)
yt = np.concatenate((yodata, yxdata), axis=0)

testx = np.concatenate((ot, xt), axis=0)
testy = np.concatenate((yxt, yot), axis=0)
# oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# wazn awaliye
weight = [i for i in np.zeros(len(dst[0]))]

wazn_nahayi = tekrar(dst, yt, weight, 0.5, 100)
yH2 = list_predict(dst, yt, wazn_nahayi)
score(yt, yH2)

# test yek tasvir  -1 baraye O      1 baraye X
test("dsox/1/X (3).jpg", wazn_nahayi)


# test
yt2 = list_predict(testx, testy, wazn_nahayi)
score(testy, yt2)
