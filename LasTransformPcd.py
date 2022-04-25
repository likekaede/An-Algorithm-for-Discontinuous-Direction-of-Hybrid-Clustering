import os
import numpy as np
import pylas
from ToolBox import  las
import open3d as o3d


def TransformToPcd(points,rgbList):
        # 传入点云对象

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(rgbList)
        o3d.visualization.draw_geometries_with_editing([pcd])
        o3d.io.write_point_cloud("E:\YeBaTanModelFiles\RightAn\\2795~2785\ModelSegment_20w.pcd", pcd)


def main():
    # filePath = pylas.read("E:\\PyProjects\\YeBanTan\\Resources\\Model.las")
    data=las.read_las("E:\YeBaTanModelFiles\RightAn\\2795~2785\ModelSegment.las")
    points=data['points']
    xList=points.x
    yList = points.y
    zList = points.z
    blueLst=points.blue
    greenLst=points.green
    redLst=points.red
    couldPoint = np.vstack((xList[0:2000000], yList[0:2000000], zList[0:2000000])).transpose()
    rgbList = np.vstack((redLst[0:2000000], greenLst[0:2000000], blueLst[0:2000000])).transpose()
    print("Successfully")
    reds=[]
    greens = []
    blues = []
    for data in rgbList:
            r = data[0]
            g = data[1]
            b = data[2]
            reds.append(r  / 255)
            greens.append(g  / 255)
            blues.append(b  / 255)
    rgbs = np.vstack((reds[0:2000000], greens[0:2000000], blues[0:2000000])).transpose()
    TransformToPcd(couldPoint,rgbs)



if __name__ == "__main__":
    main()
