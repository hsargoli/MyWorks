def hebb(data, out, w):
    wazn_nahayi = w
    for i in range(len(data)):
        wazn_marhale = []
        dws = []
        for j in range(len(data[0])):
            dw = data[i][j] * out[i]
            dws.append(dw)
            wnew = wazn_nahayi[j] + dw
            wazn_marhale.append(wnew)
        print("deltae of weights", dws)
        del dws
        wazn_nahayi = wazn_marhale
        del wazn_marhale
        print("Updated Weight =====", wazn_nahayi)

        print("Line equation:")
        print(
            "{} + {}*x1 + {}*y =0".format(
                wazn_nahayi[2], wazn_nahayi[0], wazn_nahayi[1]
            ),
            "\n\n",
        )
    return wazn_nahayi


wzn = [0, 0, 0]
# hebb for AND
data = [[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]]
output = [+1, -1, -1, -1]
w_nahayi = hebb(data, output, wzn)

print("Line equation:")
print("{}+{}*X1+{}*y=0".format(w_nahayi[2], w_nahayi[0], w_nahayi[1]))
# hebb for OR
dataor = [[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]]
outputor = [+1, +1, +1, -1]
w_nahayi2 = hebb(dataor, outputor, wzn)

print("Line equation:")
print("{}+{}*x1+{}*y=0".format(w_nahayi2[2], w_nahayi2[0], w_nahayi2[1]))
