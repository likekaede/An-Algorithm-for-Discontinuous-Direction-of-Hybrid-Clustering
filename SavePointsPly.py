import json
import numpy as np
import open3d as o3d
import  re
import sys
import math
# 求中点
def getMidPoint(first,second):
    x = (first[0] + second[0]) / 2
    y = (first[1] + second[1]) / 2
    z = (first[2] + second[2]) / 2
    return [x,y,z]
# 三维求距离
def GetDistance3d(p1, p2):
    return np.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2) + math.pow(p1[2] - p2[2], 2))
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

# 获取三维最远俩个点
def GetMinTwoPoints3d(points3d,targetPoint):
    disDic = {}
    twosPoint = []
    for p1 in points3d:
        d = GetDistance3d(p1, targetPoint)
        if not disDic.__contains__(d):
            twos = [p1]
            keyValue = {d: twos}
            disDic.update(keyValue)
    if len(disDic) > 0:
        minValue = min(disDic.keys())
        maxValue = max(disDic.keys())
        minPoint = disDic[minValue]
        maxPoint = disDic[maxValue]
        twosPoint.append(minPoint,maxPoint)
    return twosPoint

def ConvertStrToXYZ(str):
    xyz=str.split(',')
    x=(float)(xyz[0])
    y=(float)(xyz[1])
    z=(float)(xyz[2])
    return [x,y,z]


def EstimateNormals(pcd, KnnNumber=4):  # KnnNumber 表示以邻近几个点为面 不得小于3
    downpcd = pcd.voxel_down_sample(voxel_size=0.002)  # 下采样滤波，体素边长为0.002m
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=KnnNumber))  # 计算法线，只考虑邻域内的n个点
    return downpcd

def SavePlyModel(jsonPath,savePath):
    points = []
    with open(jsonPath, 'r', encoding='utf-8-sig')as fp:
        json_data = json.load(fp)
        for xyzStr in json_data:
            xyz=ConvertStrToXYZ(xyzStr)
            points.append(xyz)
        #保存节点
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(savePath, pcd)


if __name__ == '__main__':
    jsonPath = sys.argv[1]
    savePath = sys.argv[2]
    SavePlyModel(jsonPath,savePath)