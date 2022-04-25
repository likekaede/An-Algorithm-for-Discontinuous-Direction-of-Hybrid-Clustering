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

# 估算法向量
def EstimateNormals(pcd, KnnNumber=4):  # KnnNumber 表示以邻近几个点为面 不得小于3
    downpcd = pcd.voxel_down_sample(voxel_size=0.0001)  # 下采样滤波，体素边长为0.002m
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=KnnNumber))  # 计算法线，只考虑邻域内的n个点
    return downpcd

class Kmeans():
    # k: int聚类的数目
    # max_iterations: int最大迭代次数. varepsilon: float判断是否收敛
    # 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon, 则说明算法已经收敛
    def __init__(self, k=10, max_iterations=100, varepsilon=0.001):
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
        closest_i = np.argmin(distances)# 原来是argmax
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

        return self.get_cluster_labels(clusters, Data),centroids

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


def EstimateNormals(pcd, KnnNumber=6):  # KnnNumber 表示以邻近几个点为面 不得小于3
    downpcd = pcd.voxel_down_sample(voxel_size=0.002)  # 下采样滤波，体素边长为0.002m
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=KnnNumber))  # 计算法线，只考虑邻域内的n个点
    return downpcd

def EstablishModel(jsonPath):
    points = []
    pointsNew = []
    normal=[]
    with open(jsonPath, 'r', encoding='utf-8-sig')as fp:
        json_data = json.load(fp)
        for xyzStr in json_data:
            xyz=ConvertStrToXYZ(xyzStr)
            points.append(xyz)
        #从这里获取最大两个点
        twoPoints=GetFarTwoPoints3d(points)
        #从这里获取中间点
        target=getMidPoint(twoPoints[0],twoPoints[1])
        # twoPointsNew=GetMinTwoPoints3d(points,target)
        #从这里获取两个中间点形成四个点
        pointsNew.append(twoPoints[0])
        pointsNew.append(twoPoints[1])
        pointsNew.append(target)
        # pointsNew.append(twoPointsNew[0])
        # pointsNew.append(twoPointsNew[1])
        # print("123132132132132132132111111111111111111111111111")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointsNew)
        normalPcd=EstimateNormals(pcd,len(pointsNew))
        # normal= np.mean(normalPcd.normal)  # 一行解决。 normalPcd.normals[0]
        x = []
        y = []
        z = []
        for tempnor in normalPcd.normals:
            x.append(tempnor[0])
            y.append(tempnor[1])
            z.append(tempnor[2])
        xvalue = sum(x) / len(x)
        yvalue = sum(y) / len(y)
        zvalue = sum(z) / len(z)
        normal.append(xvalue)
        normal.append(yvalue)
        normal.append(zvalue)
    return normal

#使用DBscan和Skmean解译最好的normal
def DbscanAndSkmean(jsonPath):
    points = []
    normal = []
    with open(jsonPath, 'r', encoding='utf-8-sig')as fp:
        json_data = json.load(fp)
        for xyzStr in json_data:
            xyz = ConvertStrToXYZ(xyzStr)
            points.append(xyz)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # pcd = o3d.io.read_point_cloud("E:\PyProjects\YeBanTan\Resources\TestModel\Interpretation\P16.ply")
        pcd = EstimateNormals(pcd)
        xList = []
        yList = []
        zList = []
        for vector in pcd.normals:
            xList.append(vector[0])
            yList.append(vector[1])
            zList.append(vector[2])
        X = np.vstack((xList, yList, zList)).transpose()
        # 使用DBscan和Skmean解译最好的normal
        dbpcd = o3d.geometry.PointCloud()
        dbpcd.points = o3d.utility.Vector3dVector(X)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Info) as cm:
            labels = np.array(dbpcd.cluster_dbscan(eps=0.03, min_points=20, print_progress=False))
        max_label = labels.max()

        disDic = {}
        for lab in range(0,max_label):
            tempIndexes=[]
            number=0
            for index in labels:
                if lab == index and index is not -1:
                    tempIndexes.append(number)
                number=number+1

            totalCount=len(tempIndexes)
            if not disDic.__contains__(totalCount):
                keyValue = {totalCount: tempIndexes}
                disDic.update(keyValue)

        if len(disDic) > 0:
                maxValue = max(disDic.keys())
                targetIndexes = disDic[maxValue]
                xxlist = []
                yylist = []
                zzlist = []
                for index in targetIndexes:
                    point = dbpcd.points[index]
                    xxlist.append(point[0])
                    yylist.append(point[1])
                    zzlist.append(point[2])

                tempcouldPoints = np.vstack((xxlist, yylist, zzlist)).transpose()
                clf = Kmeans(k=1)
                y_pred, centroids = clf.Run(tempcouldPoints)
                normal=centroids[0]
                # print(normal)
    return normal

if __name__ == '__main__':
    jsonPath = sys.argv[1]
    # jsonPath="E:\C#Projects\SuminetAI\Project\SuminetAI\\bin\Debug\Temp\J2L-100.json"
    normal=DbscanAndSkmean(jsonPath)
    if len(normal)>0:
        print(str(normal[0])+','+str(normal[1])+','+str(normal[2]))
    else:
        print(str(0) + ',' + str(0) + ',' + str(1))