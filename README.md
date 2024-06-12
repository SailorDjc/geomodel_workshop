# geomodel_workshop

## 依赖环境安装说明
* pytorch
* dgl
* pyvista
* vtk
* scikit-learn
* pandas
* torchmetrics
* matplotlib
* xgboost
* pytest
* pynoddy
* imageio
* rdp
* openpyxl<br />

注1：本项目使用的是python3.8环境，cuda11.6，
1. pytorch安装命令为：conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia <br />
2. dgl安装命令为：conda install -c dglteam/label/cu116 dgl<br />
3. 其他依赖项直接用pip install 依次安装即可。<br />

注2：本项目引入了noddy地质模型数据集，引用了pynoddy库进行地质模拟模型的构建，该库的安装步骤需要做以下说明：
1. pip install pynoddy  安装pynoddy库<br />
2. 下载noddy app可执行文件，下载网址是 http://www.tectonique.net/pynoddy 或在 [pynoddy的github仓库下](https://github.com/flohorovicic/pynoddy/tree/master/noddyapp)
     选择适配自己计算机的程序版本，将可执行文件重命名为 noddy.exe ， 将其所在目录添加到计算机环境变量Path中。 <br />
* 引入pytetgen <br />
由于scipy.spatial提供的Delaunay三角网生成算法在面对大规模点云网格生成时出现计算性能瓶颈的问题，所以考虑引入pytetgen库的三角网算法，直接安装:pip install pytetgen可能会出现问题，<br />
因为这个库需要Cython编译，而编译过程中会遇到python2与python3冲突的问题，python2允许将doulbe类型值赋值给int类型的变量，而python中不允许，所以安装方法如下：<br />
```
pip install Cython==0.29.35  # 该版本支持python2的规则
pip install pytetgen
```
注3：项目添加了地形曲面约束的模块，允许输入DEM的tiff文件生成地形面，因此需要引入rasterio库。
* 安装rasterio [optional]
依次安装 * pyproj Shapely GDAL Fiona geopandas rasterio 
其中GDAL,Fiona,rasterio不可以用pip install安装，需要下载whl安装包进行离线安装，下载网址 https://www.lfd.uci.edu/~gohlke/pythonlibs/ 或 https://www.cgohlke.com/
在选择whl包时，建议选择指定python的最高版本库，且注意包后缀为cp，意为使用的CPython实现，例如GDAL-3.4.3-cp38-cp38-win_amd64.whl。
## 代码运行说明

