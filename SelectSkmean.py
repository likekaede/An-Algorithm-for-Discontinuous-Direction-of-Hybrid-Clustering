import json
import numpy as np
import open3d as o3d
import  re
import sys
import math

# 计算一个样本与数据集中所有样本的欧氏距离的平方
def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    X = X.reshape(X.shape[0], -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances


# 计算一个样本与数据集中所有样本的余眩度
def euclidean_Cos(sample, vector2):
    vector2 = vector2.reshape(vector2.shape[0], -1)
    coses = []
    for d in vector2:
        a2 = math.sqrt(sample[0] * sample[0] + sample[1] * sample[1] + sample[2] * sample[2])
        b2 = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        down = a2 * b2
        up = sample[0] * d[0] + sample[1] * d[1] + sample[2] * d[2]
        result = up / down
        coses.append(result)
    # print(str(result))
    return coses

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

#获取所有点
def GetPoint3dPcd(jsonPath):
    points = []
    f = open(jsonPath, "r", encoding='utf-8-sig')
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
    for xyzStr in lines:
        if xyzStr=="\n":
            continue
        xyz = ConvertStrToXYZ(xyzStr)
        points.append(xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


class SKmeans():
    # k: int聚类的数目
    # max_iterations: int最大迭代次数. varepsilon: float判断是否收敛
    # 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon, 则说明算法已经收敛
    def __init__(self, k=3, max_iterations=200, varepsilon=0.001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon

    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, Data):
        n_samples, n_features = np.shape(Data)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = Data[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # 返回距离该样本最近的一个中心索引[0, self.k)
    def closest_centroid(self, sample, centroids):
        # distances = euclidean_distance(sample, centroids) #更新样本用  距离
        distances = euclidean_Cos(sample, centroids) # 更新样本用余玄度
        closest_i = np.argmax(distances)
        return closest_i

    # 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
    def create_clusters(self, centroids, Data):
        # n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(Data):
            centroid_i = self.closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    # 对中心进行更新
    def update_centroids(self, clusters, Data):
        n_features = np.shape(Data)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(Data[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # 将所有样本进行归类，其所在的类别的索引就是其类别标签
    def get_cluster_labels(self, clusters, Data):
        y_pred = np.zeros(np.shape(Data)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # 对整个数据集进行Kmeans聚类，返回其聚类的标签
    def Run(self, Data):
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        centroids = self.init_random_centroids(Data)
        # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
        for n in range(self.max_iterations):
            # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
            clusters = self.create_clusters(centroids, Data)
            former_centroids = centroids
            # 计算新的聚类中心
            centroids = self.update_centroids(clusters, Data)
            # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break

        return self.get_cluster_labels(clusters, Data)

if __name__ == '__main__':
    jsonPath = sys.argv[1] # 原始点json
    # jsonPath="E:\C#Projects\SuminetAI\Project\SuminetAI\\bin\Debug\Temp\TempDisconPoints.json"
    knnnumber = sys.argv[2]# skmean的算法 算子
    outputJsonPath= sys.argv[3]#  输出的json
    # outputJsonPath="E:\C#Projects\SuminetAI\Project\SuminetAI\\bin\Debug\Temp\TempDisconPoints_output.json"
    # knnnumber=3
    knnnumber=(int)(knnnumber)
    # outputJsonPath = sys.argv[3] #输出的json路径
    pcd=GetPoint3dPcd(jsonPath)
    pcd = EstimateNormals(pcd)
    xList = []
    yList = []
    zList = []
    xListp = []
    yListp = []
    zListp = []
    for vector in pcd.normals:
        xList.append(vector[0])
        yList.append(vector[1])
        zList.append(vector[2])
    for point in pcd.points:
        xListp.append(point[0])
        yListp.append(point[1])
        zListp.append(point[2])
    X = np.vstack((xList, yList, zList)).transpose()
    X2 = np.vstack((xListp, yListp, zListp)).transpose()
    # 用Kmeans算法进行聚类
    clf = SKmeans(k=knnnumber)
    y_pred = clf.Run(X)
    # 可视化聚类效果
    # pcds = []
    skemenJson=[]
    for n in range(0, knnnumber):
        tempx = X2[y_pred == n][:, 0]
        tempy = X2[y_pred == n][:, 1]
        tempz = X2[y_pred == n][:, 2]
        tempxV = X[y_pred == n][:, 0]
        tempyV = X[y_pred == n][:, 1]
        tempzV = X[y_pred == n][:, 2]

        temppcd = o3d.geometry.PointCloud()
        tempcouldPoint = np.vstack((tempx, tempy, tempz)).transpose()
        a = np.random.random()
        b = np.random.random()
        c = np.random.random()

        # temppcd.points = o3d.utility.Vector3dVector(tempcouldPoint)
        # temppcd.paint_uniform_color([a, b, c])  # 把所有点渲染为灰色
        # print([a, b, c])
        # pcds.append(temppcd)

        dicJson = []
        for xvalue,yvalue,zvalue,xvalueV,yvalueV,zvalueV  in zip(tempx,tempy,tempz,tempxV,tempyV,tempzV):
            xyzJson = { "X": xvalue, "Y": yvalue, "Z": zvalue,"XV": xvalueV, "YV": yvalueV, "ZV": zvalueV,}
            dicJson.append(xyzJson)
        skemenJson.append(dicJson)
        #每一次
    json_str = json.dumps(skemenJson)
    with open(outputJsonPath, 'w') as json_file:
        json_file.write(json_str)

    # o3d.visualization.draw_geometries(pcds)可视化

    # indexes = vis.get_picked_points()
    # dicJson = []
    # if len(indexes) == 1:
    #     index = indexes[0]
    #     normal = pcd.normals[index]
    #     print(str(normal[0]) + ',' + str(normal[1]) + ',' + str(normal[2]))
    #
        #
        # temppcd = o3d.geometry.PointCloud()
        # tempcouldPoint = np.vstack((tempx, tempy, tempz)).transpose()
        # a = np.random.random()
        # b = np.random.random()
        # c = np.random.random()
        # temppcd.points = o3d.utility.Vector3dVector(tempcouldPoint)
        # temppcd.paint_uniform_color([a, b, c])  # 把所有点渲染为灰色
        # pcds.append(temppcd)
