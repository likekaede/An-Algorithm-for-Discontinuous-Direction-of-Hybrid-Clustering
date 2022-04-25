import math
import cv2
import json
import numpy as np
import open3d as o3d
import sys


def Normalization(nd, x):
    x = np.asarray(x)
    m, s = np.mean(x, 0), np.std(x)
    if nd == 2:
        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])

    Tr = np.linalg.inv(Tr)
    x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
    x = x[0:nd, :].T

    return Tr, x


def DLTcalib(nd, xyz, uv):
    '''
     DLT 使用已知对象点及其图像点进行相机校准。
     输入
    -----
     nd：对象空间的维度，此处为 3。
     xyz：对象 3D 空间中的坐标。
     uv：图像二维空间中的坐标。
     坐标 (x,y,z 和 u,v) 以列的形式给出，不同的点以行的形式给出。
     3D DLT 必须至少有 6 个校准点。
     输出
    ------
     L：校准矩阵的 11 个参数的数组。
    err：DLT 的误差（以相机坐标为单位的 DLT 变换的平均残差）。
    '''
    if nd != 3:
        raise ValueError('%dD DLT unsupported.' % (nd))

    # 转换所有的变量成为numpy列 Converting all variables to numpy array
    xyz = np.asarray(xyz)
    uv = np.asarray(uv)
    n = xyz.shape[0]
    # Validating the parameters:
    if uv.shape[0] != n:
        raise ValueError('Object (%d points) and image (%d points) have different number of points.' % (n, uv.shape[0]))
    if xyz.shape[1] != 3:
        raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).' % (xyz.shape[1], nd, nd))
    if n < 6:
        raise ValueError(
            '%dD DLT requires at least %d calibration points. Only %d points were entered.' % (nd, 2 * nd, n))
    # 规范化数据以提高 DLT 质量（DLT 取决于坐标系）。
    # 当存在相当大的透视失真时，这是相关的。
    # 归一化：原点的平均位置和平均距离在每个方向都等于 1。
    Txyz, xyzn = Normalization(nd, xyz)
    Tuv, uvn = Normalization(2, uv)
    A = []
    for i in range(n):
        x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
        u, v = uvn[i, 0], uvn[i, 1]
        A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])

    # 将 A 转换为数组
    A = np.asarray(A)

    # 找到11个参数：
    U, S, V = np.linalg.svd(A)

    # 参数在 Vh 的最后一行，并对其进行归一化
    L = V[-1, :] / V[-1, -1]
    # print(L)
    # 相机投影矩阵
    H = L.reshape(3, nd + 1)
    # print(H)

    # 反归一化
    # pinv:矩阵的Moore-Penrose伪逆
    H = np.dot(np.dot(np.linalg.pinv(Tuv), H), Txyz)
    # print(H)
    H = H / H[-1, -1]
    # print(H)
    L = H.flatten('A')
    # print(L)

    # DLT的平均误差（DLT变换的平均残差，单位为摄像机坐标）
    uv2 = np.dot(H, np.concatenate((xyz.T, np.ones((1, xyz.shape[0])))))
    uv2 = uv2 / uv2[2, :]
    # 平均距离：
    err = np.sqrt(np.mean(np.sum((uv2[0:2, :].T - uv) ** 2, 1)))

    return L, err


# 开始计算Dlt矩阵
def Dlt(xyz_in, uv_in):
    xyz = xyz_in
    uv = uv_in
    nd = 3
    return DLTcalib(nd, xyz, uv)


# 映射XYZ 到 UV
def Mapping(X, Y, Z, matrix):
    L1 = matrix[0]
    L2 = matrix[1]
    L3 = matrix[2]
    L4 = matrix[3]
    L5 = matrix[4]
    L6 = matrix[5]
    L7 = matrix[6]
    L8 = matrix[7]
    L9 = matrix[8]
    L10 = matrix[9]
    L11 = matrix[10]
    u = -(L1 * X + L2 * Y + L3 * Z + L4) / (L9 * X + L10 * Y + L11 * Z + 1)
    v = -(L5 * X + L6 * Y + L7 * Z + L8) / (L9 * X + L10 * Y + L11 * Z + 1)
    return int(math.fabs(u)), int(math.fabs(v))


# 获取2d数据
def GetSixPoint2d(sixPoint2dPath):
    sixPoint2ds = []
    with open(sixPoint2dPath, 'r', encoding='utf-8-sig')as fp:
        json_data = json.load(fp)
    for temPoint2d in json_data:
        point2dList = []
        point2dList.append(temPoint2d["U"])
        point2dList.append(temPoint2d["V"])
        sixPoint2ds.append(point2dList)
    return sixPoint2ds


# 获取3d数据
def GetSixPoint3d(sixPoint3dPath):
    sixPoint3ds = []
    with open(sixPoint3dPath, 'r', encoding='utf-8')as fp:
        json_data = json.load(fp)
    for temPoint3d in json_data:
        point3dList = []
        point3dList.append(temPoint3d["x"])
        point3dList.append(temPoint3d["y"])
        point3dList.append(temPoint3d["z"])
        sixPoint3ds.append(point3dList)
    return sixPoint3ds


def StartMap(matrix, pcdFile, imageFile, outMapFile, outImageMaskFile):
    pcd = o3d.io.read_point_cloud(pcdFile)
    img = cv2.imread(imageFile)
    height = img.shape[0]
    width = img.shape[1]
    img_mask = np.zeros((height, width, 3), np.uint8)  # 复制一张空白的背景图片
    img_mask.fill(255)  # 使用白色填充图片区域,默认为黑色
    # 写文件句柄
    handle = open(outMapFile, 'a')
    handle.write('# Utf-8\n')
    handle.write('Version v1.0\n')
    handle.write('#.smap format u v x y z r g b\n')
    handle.write('Points Count ' + str(len(pcd.points)) + "\n")
    for point, color in zip(pcd.points, pcd.colors):
        x = point[0]
        y = point[1]
        z = point[2]
        r = color[0] * 255
        g = color[1] * 255
        b = color[2] * 255
        u, v = Mapping(x, y, z, matrix)
        if u >= width:
            continue
        if v >= height:
            continue
        string = str(u) + "," + str(v) + "," + str(round(x, 2)) + "," + str(round(y, 2)) + "," + str(
            round(z, 2)) + "," + str(int(r)) + "," + str(int(g)) + "," + str(int(b)) + "\n"
        handle.write(string)
        color = [b, g, r]
        point = (u, v)
        cv2.circle(img_mask, point, 3, color, 4)
        # cv2.rectangle()
    handle.close()
    cv2.imwrite(outImageMaskFile, img_mask)


if __name__ == '__main__':
    six2dPath = sys.argv[1]
    six3dPath = sys.argv[2]
    pcdPath = sys.argv[3]
    smapPath = sys.argv[4]
    originImagePath = sys.argv[5]
    outImagePath = sys.argv[6]
    # six2dPath = "E:\PyProjects\YeBanTan\Resources\TestModel\SixPoint2d.json"
    # six3dPath = "E:\PyProjects\YeBanTan\Resources\TestModel\SixPoint3d.json"
    # pcdPath = "E:\PyProjects\YeBanTan\Resources\TestModel\cropped_2.ply"
    # smapPath = "E:\PyProjects\YeBanTan\Resources\TestModel\map76.smap"
    # originImagePath = "E:\PyProjects\YeBanTan\Resources\TestModel\\1-1-2.jpg"
    # outImagePath ="E:\PyProjects\YeBanTan\Resources\TestModel\\imageCould.jpg"
    Match2ds = GetSixPoint2d(six2dPath)
    Match3ds = GetSixPoint3d(six3dPath)
    matrix, error = Dlt(Match3ds, Match2ds)
    print("配准残差：" + str(round(error,2)))
    StartMap(matrix, pcdPath, originImagePath, smapPath, outImagePath)
