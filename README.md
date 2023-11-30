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
