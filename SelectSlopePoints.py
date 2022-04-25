import numpy as np
import pylas
import open3d as o3d
import json
import sys

# 选择六个三维点云配准点

def PickPoints(pcdFile, fourJsonFile):
    pcd = o3d.io.read_point_cloud(pcdFile)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window("选择四个边坡角点", width=1200, height=680)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    indexes = vis.get_picked_points()
    temJson = {"Error": "The Point Number is lower Four "}
    dicJson = []
    newPoints=[]
    if len(indexes) == 4:
        for i in range(4):
            index = indexes[i]
            point = pcd.points[index]
            newPoints.append(point)
            temJson = {"Id": i+1, "x": point[0], "y": point[1], "z": point[2]}
            dicJson.append(temJson)
    else:
        dicJson.append(temJson)

    newpcd = o3d.geometry.PointCloud()
    newpcd.points = o3d.utility.Vector3dVector(newPoints)
    downpcd = newpcd.voxel_down_sample(voxel_size=0.002)
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=len(indexes)))
    x=[]
    y=[]
    z=[]
    for  tempnor in downpcd.normals:
        x.append(tempnor[0])
        y.append(tempnor[1])
        z.append(tempnor[2])
    xvalue=sum(x) / len(x)
    yvalue = sum(y) / len(y)
    zvalue = sum(z) / len(z)

    normalJson = {"Id": 5, "x": xvalue, "y": yvalue, "z": zvalue}
    dicJson.append(normalJson)

    json_str = json.dumps(dicJson)
    with open(fourJsonFile, 'w') as json_file:
        json_file.write(json_str)
    return True


if __name__ == '__main__':
    pcdPath = sys.argv[1]
    saveJsonPath = sys.argv[2]
    PickPoints(pcdPath, saveJsonPath)

