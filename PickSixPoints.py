import numpy as np
import pylas
import open3d as o3d
import json
import sys

# 选择六个三维点云配准点
def PickPoints(pcdFile, sixJsonFile):
    pcd = o3d.io.read_point_cloud(pcdFile)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window("选择六个配准点", width=1200, height=680)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    indexes = vis.get_picked_points()
    temJson = {"Error": "The Point Number is lower six "}
    dicJson = []
    if len(indexes) == 6:
        for i in range(6):
            index = indexes[i]
            point = pcd.points[index]
            temJson = {"Id": i+1, "x": point[0], "y": point[1], "z": point[2]}
            dicJson.append(temJson)
    else:
        dicJson.append(temJson)
    json_str = json.dumps(dicJson)
    with open(sixJsonFile, 'w') as json_file:
        json_file.write(json_str)
    return True


def TransformToPcd(lazFilePath, pcdFilePath, couldNumber=3000000):
    laz = pylas.read(lazFilePath)
    length = len(laz.points)
    if couldNumber >= length:
        return False
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
    return True


if __name__ == '__main__':
    lazPath = sys.argv[1]
    pcdPath = sys.argv[2]
    saveJsonPath = sys.argv[3]
    isHavePcd = sys.argv[4]
    if isHavePcd=='Yes':
        PickPoints(pcdPath, saveJsonPath)
    else:
        TransformToPcd(lazPath, pcdPath)
        PickPoints(pcdPath, saveJsonPath)

