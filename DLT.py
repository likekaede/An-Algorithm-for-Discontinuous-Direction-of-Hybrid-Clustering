import numpy as np
import json


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


def Run(xyz_in, uv_in, outMatrixJsonPath):  # 开始计算
    xyz = xyz_in
    uv = uv_in
    nd = 3
    matrix, error = DLTcalib(nd, xyz, uv)
    dicJson = []
    for i in range(12):
        dic = {'L' + str(i): matrix[i]}
        dicJson.append(dic)
    dicError = {"error": error}
    dicJson.append(dicError)
    json_str = json.dumps(dicJson)
    with open(outMatrixJsonPath, 'w') as json_file:
        json_file.write(json_str)

# if __name__ == '__main__':
#     # Known 3D coordinates   已知三维空间点
#     xyz = [[1.27877855e+01, -2.75449705e+00, 2.81439453e+03], [6.42778492, -9.64349747, 2812.42163086],
#            [3.55978513, -8.58049679, 2809.29052734],
#            [8.33878517, -5.31049728, 2811.35449219], [7.36778498, -3.64749694, 2809.50048828],
#            [1.02667847e+01, -1.29049706e+00, 2.81132349e+03]]
#     # Known pixel coordinates 对应的二维相片像素点
#     uv = [[796.2,378], [4197.4,1281], [4542,2992.2], [2423.6,1773.8], [2061.4,2731.2], [707.44,1887.7]]
#     Run(xyz,uv,"E:\PyProjects\YeBanTan\Resources\matrix_data.json")
