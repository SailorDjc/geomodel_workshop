# geomodel_workshop
## 依赖环境
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
* pynoddy <br /><br />
注：本项目引入了noddy地质模型数据集，引用了pynoddy库进行地质模拟模型的构建，该库的安装步骤需要做以下说明：
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
