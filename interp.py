from scipy import interpolate
import matplotlib.pyplot as plt
import csv
import os


def interp1(x, y, z, interpz):
    # interplinex = interpolate.interp1d(z, x, kind='cubic')
    # interpliney = interpolate.interp1d(z, y, kind='cubic')
    interplinex = interpolate.InterpolatedUnivariateSpline(z, x, k=2)
    interpliney = interpolate.PchipInterpolator(z, y)
    # interpliney = interpolate.InterpolatedUnivariateSpline(z, y, k=3)
    interpx = interplinex(interpz)
    interpy = interpliney(interpz)
    return interpx, interpy


def sort3(s):
    return s[2]


if __name__ == '__main__':
    cls_root = 'E:\\test\\csv\\'
    csv_root = 'E:\\test\\markercsv\\'
    interpcsvlist = os.listdir(csv_root)
    ex = []
    ey = []
    e = []
    for patientid1 in range(len(interpcsvlist)):
        csv_path = os.path.join(csv_root, interpcsvlist[patientid1])
        markups = []
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                row = [float(row[0]), float(row[1]), float(row[2])]
                markups.append(row)
        markups.sort(key=sort3)
        realx = [float(temp[0]) for temp in markups]
        realy = [float(temp[1]) for temp in markups]
        interpz = [float(temp[2]) for temp in markups]
        x = []
        y = []
        z = []
        for i in range(0, len(interpz), int(len(interpz)/20 + 1)):
            x.append(realx[i])
            y.append(realy[i])
            z.append(interpz[i])
        if z[-1] != interpz[-1] and len(z) - i <= 3:
            x.append(realx[-1])
            y.append(realy[-1])
            z.append(interpz[-1])
        interpx, interpy = interp1(x, y, z, interpz)
        dx = abs(interpx - realx)
        dy = abs(interpy - realy)
        ex.append(max(dx))
        ey.append(max(dy))
        dz = []
        for i in range(len(interpz)):
            dz.append(dx[i] ** 2 + dy[i] ** 2)
        e.append(sum(dz) ** 0.5)
        plt.figure()
        plt.subplot(121)
        plt.plot(interpx, interpz, 'o')
        plt.subplot(122)
        plt.plot(interpy, interpz, 'o')
        plt.show()
    print(interpcsvlist)
    print(ex, '\n', ey, '\n', e)
    exit()
