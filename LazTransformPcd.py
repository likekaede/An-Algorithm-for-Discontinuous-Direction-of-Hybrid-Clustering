import numpy as np
import pylas
import open3d as o3d

# 转换Laz文件到Pcd文件格式


def TransformToPcd(lazFilePath, pcdFilePath, couldNumber):
    laz = pylas.read(lazFilePath)
    length = len(laz.points)
    if couldNumber >= length:
        return None
    couldPoint = np.vstack((laz.x[0:couldNumber], laz.y[0:couldNumber], laz.z[0:couldNumber])).transpose()
    points = laz.points
    blueLst = []
    greenLst = []
    redLst = []
    for data in points[0:couldNumber]:
        r = data[9]
        g = data[10]
        b = data[11]
        redLst.append(r / 256 / 255)
        greenLst.append(g / 256 / 255)
        blueLst.append(b / 256 / 255)
    rgbList = np.vstack((redLst[0:couldNumber], greenLst[0:couldNumber], blueLst[0:couldNumber])).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(couldPoint)
    pcd.colors = o3d.utility.Vector3dVector(rgbList)
    o3d.io.write_point_cloud(pcdFilePath, pcd)
if __name__ == '__main__':
    TransformToPcd("E:\\叶巴滩模型文件\\右岸\\2815~2805\\123.laz","E:\\叶巴滩模型文件\\右岸\\2815~2805\\123.pcd",10000000)