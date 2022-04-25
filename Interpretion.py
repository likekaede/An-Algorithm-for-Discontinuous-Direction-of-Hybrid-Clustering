import json
import cv2
import open3d as o3d
import numpy as np
import math
# 判断一个点是否在多边形以内2d
def IsInnerPoint2d(p, poly):
    isIn = False
    px = p[0]
    py = p[1]
    flag = False
    i = 0
    l = len(poly)
    j = l - 1
    while i < l:
        sx = poly[i][0]
        sy = poly[i][1]
        tx = poly[j][0]
        ty = poly[j][1]
        # 点与多边形顶点重合
        if (sx == px and sy == py) or (tx == px and ty == py):
            return (px, py)
        # 判断线段两端点是否在射线两侧
        if (sy < py and ty >= py) or (sy >= py and ty < py):
            # 线段上与射线 Y 坐标相同的点的 X 坐标
            x = sx + (py - sy) * (tx - sx) / (ty - sy)
            # 点在多边形的边上
            if x == px:
                return (px, py)
            # 射线穿过多边形的边界
            if x > px:
                flag = not flag
        j = i
        i += 1
    # 射线穿过多边形边界的次数为奇数时点在多边形内
    if flag:
        isIn = True
    return isIn
# 三维求距离
def GetDistance3d(p1, p2):
    return np.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2) + math.pow(p1[2] - p2[2], 2))
# 二维求距离
def GetDistance2d(p1, p2):
    return np.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))
# 获取三维最远俩个点
def GetFarTwoPoints3d(points3d):
    disDic = {}
    twosPoint = []
    for p1 in points3d:
        for p2 in points3d:
            d = GetDistance3d(p1, p2)
            if not disDic.__contains__(d):
                twos = [p1, p2]
                keyValue = {d: twos}
                disDic.update(keyValue)
    if len(disDic) > 0:
        maxValue = max(disDic.keys())
        twosPoint = disDic[maxValue]
    return twosPoint

# 获取二维最远俩个点
def GetFarTwoPoints2d(points2d):
    disDic = {}
    twosPoint = []
    for p1 in points2d:
        for p2 in points2d:
            d = GetDistance2d(p1, p2)
            if not disDic.__contains__(d):
                twos = [p1, p2]
                keyValue = {d: twos}
                disDic.update(keyValue)
    if len(disDic) > 0:
        maxValue = max(disDic.keys())
        twosPoint = disDic[maxValue]
    x1 = int(twosPoint[0][0])
    y1 = int(twosPoint[0][1])
    x2 = int(twosPoint[1][0])
    y2 = int(twosPoint[1][1])
    ps = (x1, y1)
    pe = (x2, y2)
    return ps, pe
# 从所有二维数据寻找二维多边形里面对应的所有三维点
def GetInnerPoints(uvs, xyzs, poly2d):
    isInnerPoints = []
    isInnerIndexs = []
    i = 0
    for uv in uvs:
        isInner = IsInnerPoint2d(uv, poly2d)
        if isInner:
            isInnerPoints.append(uv)
            isInnerIndexs.append(i)
        i = i + 1
    ploys3d = []
    for index in isInnerIndexs:
        xyz = xyzs[index]
        ploys3d.append(xyz)
    return ploys3d
# 读取映射文件
def ReadSmapInfo(smapFilePath):
    uvs=[]
    xyzs=[]
    with open(smapFilePath, "r") as f:
        i = 0
        for line in f.readlines():
            i = i + 1
            line = line.strip('\n')
            if i <= 4:
                continue
            else:
                uv_xyz_rgb = line.split(',')
                u = uv_xyz_rgb[0]
                v = uv_xyz_rgb[1]
                x = uv_xyz_rgb[2]
                y = uv_xyz_rgb[3]
                z = uv_xyz_rgb[4]
                uv = [int(u), int(v)]
                xyz = [float(x), float(y), float(z)]
                uvs.append(uv)
                xyzs.append(xyz)
    return uvs,xyzs
#读取掩码Json文件(返回shapes数据)
def ReadMaskJson(maskJsonPath):
    data = json.load(open(maskJsonPath, encoding="utf-8"))
    return data['shapes']
#根据法向量解译倾向、倾角
def InterpretDipS(tempNor):
    l=tempNor[0]
    m=tempNor[1]
    n=tempNor[2]
    beita=math.acos(n)
    sinBeita=math.sin(beita)
    temP=m/sinBeita
    aref=math.asin(temP)
    dipDirection=aref/ math.pi * 180
    dipAngle=beita/ math.pi * 180
    return dipDirection,dipAngle
