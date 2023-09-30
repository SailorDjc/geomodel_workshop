import random
import tarfile

import numpy as np
import pyvista as pv
import pynoddy.output
import pynoddy.history
import os
import gzip
import time
from tqdm import tqdm
import pickle
import copy
import requests
from contextlib import closing
import hashlib
from urllib.request import urlopen
import random
import time
import shutil


class NoddyModelData(object):
    def __init__(self, root=None, save_dir_name='model_data', dataset_list=None, max_model_num=None):
        if max_model_num is None:  # 生成模型的个数
            max_model_num = 100
        if dataset_list is None or len(dataset_list) == 0:
            dataset_list = ['FOLD_FOLD_FOLD']
        self.cur_dataset = dataset_list[0]
        self.root = root  # 数据集根目录
        self.raw_dir_path = os.path.join(root, 'data', 'raw_data')  # raw_data: 存放tar格式数据集
        self.his_dir_path = os.path.join(root, 'data', 'his_dir')  # 按模型分文件夹存放，.his .grv .g00 .g12 .mag 等文件
        # 模型格网文件 .vtr 格式，可以直接读取，文件夹中建立模型列表txt文件
        self.output_dir = os.path.join(root, 'data', save_dir_name)
        # 数据日志记录文件
        # dataset - model_files  记录有哪些数据集  {dataset:[model_name]}
        self.dataset_list_path = os.path.join(root, 'data', 'dataset_list_log.pkl')  # 记录有哪些模型文件数据集
        # model_files-file_path  记录每个模型对应的his文件地址  {model_name:his_file_path}
        self.model_his_list_path = os.path.join(root, 'data', 'model_his_list_log.pkl')  # 记录每一个模型his文件的地址列表
        # vtr_model_list  记录每个dataset存储了哪些vtk_model  {dataset:[model_name]} 形式上与dataset_list_log一致
        self.grid_model_list_path = os.path.join(root, 'data', 'grid_model_list_log.pkl')  # 记录已生成的格网模型列表
        # noddy model grid param   记录了模型的基本参数 {model_name:[nx, ny, nz, cell_x, cell_y, cell_z, extent_x,
        #                                                                  extent_y, extent_z]}
        self.model_param_list_path = os.path.join(root, 'data', 'model_param_list_log.pkl')  # noddy格网模型参数
        # 加载数据集元数据
        self.dataset_list_log, self.model_his_list_log, self.grid_model_list_log, self.model_param_list_log = \
            self.load_log_files()

        # 检查文件夹是否存在，不存在就新建
        if not os.path.exists(self.his_dir_path):
            os.makedirs(self.his_dir_path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.raw_dir_path):
            os.makedirs(self.raw_dir_path)

        self.dataset_list = dataset_list  # 指定加载的数据集
        # 会先到 raw_data文件夹中搜索，搜索不到就下载
        saved_tar_files = self.get_raw_dataset_list()
        if self.dataset_list is None:
            # 如果 dataset_list为空，则默认将raw_data文件夹中的所有数据集加载
            # 检查数据集列表记录，与模型列表记录匹配
            self.dataset_list = saved_tar_files
        else:
            # 如果指定数据集不存在，则下载，若无法下载，则忽略
            for dataset_name in self.dataset_list:
                if dataset_name not in saved_tar_files:
                    download_flag = self.download_dataset(dataset=dataset_name)
                    if download_flag is False:
                        continue
        self.extract_his_files_from_tar_files(des_dir=self.his_dir_path)
        self.generate_model_by_his(self.output_dir, max_num=max_model_num)

        saved_files = self.get_all_noddy_model_names()
        # 删除已经存储模型的中间文件和压缩包，节约存储空间
        self.delete_gz_files()
        self.delete_extra_model_files(dir_path=self.output_dir, file_names=saved_files)
        self.delete_extra_his_files(dir_path=self.his_dir_path, file_names=saved_files)

    # 在线下载原始数据集
    def download_dataset(self, dataset):
        his_filter = dataset.split('_')
        if len(his_filter) != 3:
            print("Geological events need to have three.")
            return False
        url = "https://cloudstor.aarnet.edu.au/plus/s/UxnVSkHfnr7chW9/download?path=%2F&files="
        path = url + dataset + '.tar'
        start_time = time.time()
        filepath = os.path.join(self.raw_dir_path, dataset + '.tar')
        try:
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                              ' (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}
            r = requests.get(path, stream=True, headers=headers, timeout=3)
            size = 0  # 初始化已下载大小
            chunk_size = 1024  # 每次下载的数据大小
            content_size = int(r.headers['content-length'])  # 下载文件总大小
            if r.status_code == 200:  # 判断是否响应成功
                # 开始下载，显示下载文件大小
                print('Start download, [File size]:{size:.2f} MB'.format(size=content_size / chunk_size))
                with open(filepath, 'wb') as f:  # 显示进度条
                    for data in r.iter_content(chunk_size=chunk_size):
                        f.write(data)
                        size += len(data)
                        print('\r' + '[Download progress]:%s%.2f%%' %
                              ('>' * int(size * 50 / content_size), float(size / content_size * 100)), end=' ')
        except:
            print('open url error, cannot download dataset {}'.format(dataset))
            return False

    def load_log_files(self):
        if os.path.exists(self.dataset_list_path):
            dataset_list_log = self.load_data(self.dataset_list_path)
        else:
            dataset_list_log = dict()
        if os.path.exists(self.model_his_list_path):
            model_his_list_log = self.load_data(self.model_his_list_path)
        else:
            model_his_list_log = dict()
        if os.path.exists(self.grid_model_list_path):
            grid_model_list_log = self.load_data(self.grid_model_list_path)
        else:
            grid_model_list_log = dict()
        if os.path.exists(self.model_param_list_path):
            model_param_list_log = self.load_data(self.model_param_list_path)
        else:
            model_param_list_log = dict()
        return dataset_list_log, model_his_list_log, grid_model_list_log, model_param_list_log

    def save_data(self, save_path, data):
        out_put = open(save_path, 'wb')
        out_str = pickle.dumps(data)
        out_put.write(out_str)
        out_put.close()

    def load_data(self, load_path):
        with open(load_path, 'rb') as file:
            geodata = pickle.loads(file.read())
            return geodata

    # 获取已有数据集名字
    def get_raw_dataset_list(self):
        tar_files = os.listdir(self.raw_dir_path)
        data_list = []
        for tar_file in tar_files:
            if tar_file.endswith('.tar'):
                file = tar_file.replace('.tar', '')
                data_list.append(file)
        data_list = list(set(data_list))
        return data_list

    # 获取所有noddy模型名
    def get_all_noddy_model_names(self):
        all_model_list = []
        for dataset in self.grid_model_list_log.keys():
            model_list = self.grid_model_list_log[dataset]
            all_model_list.extend(model_list)
        return all_model_list

    def get_noddy_model_list_names(self, dataset='FOLD_FOLD_FOLD', model_num: int = -1, sample_random=False):
        if dataset not in self.grid_model_list_log.keys():
            return []
        if model_num <= 0 or model_num > self.get_model_num(dataset=dataset):
            model_num = self.get_model_num(dataset=dataset)
        model_list = self.grid_model_list_log[dataset]
        if sample_random is True:
            result = random.sample(model_list, model_num)
            return result
        else:
            return model_list[0:model_num]

    def get_noddy_model_list_paths(self, dataset='FOLD_FOLD_FOLD', model_num=100, sample_random=False):
        model_list = self.get_noddy_model_list_names(dataset=dataset, model_num=model_num, sample_random=sample_random)
        path_list = [os.path.join(self.output_dir, dataset, model_name) for model_name in model_list]
        return path_list

    # return dict {model_name: model_path}
    def get_noddy_model_list_path_by_names(self, model_name_list):
        model_list_path = {}
        for model_name in model_name_list:
            dn = None
            if model_name not in model_list_path.keys():
                model_list_path[model_name] = None
            for dataset in self.grid_model_list_log.keys():
                if model_name in self.grid_model_list_log[dataset]:
                    dn = dataset
                    break
            if dn is not None:
                model_list_path[model_name] = os.path.join(self.output_dir, dn, model_name)
        return model_list_path

    def get_grid_model(self, model_name):
        model_paths = self.get_noddy_model_list_path_by_names([model_name])
        if model_paths is not None:
            model_path = model_paths[model_name]
            bin_path = model_path + '.pkl'
            grid_model = self.load_data(bin_path)
            return grid_model
        else:
            return None

    def get_grid_model_by_idx(self, dataset:str, idx: [int]):
        model_num = self.get_model_num(dataset=dataset)
        model_names = self.get_noddy_model_list_names(dataset=dataset)
        if isinstance(idx, list):
            model_names = [model_names[id] for id in idx if 0 <= id < model_num]
            grid_model_path_map = self.get_noddy_model_list_path_by_names(model_names)
            grid_models = []
            for model_name in model_names:
                if model_name in grid_model_path_map.keys():
                    bin_path = grid_model_path_map[model_name] + '.pkl'
                    grid_model = self.load_data(bin_path)
                    grid_models.append(grid_model)
            return grid_models
        else:
            print('idx is not list type.')
            return []

    # dataset='FOLD_FOLD_FOLD'
    def get_model_num(self, dataset=None):
        if dataset is not None:
            if dataset in self.grid_model_list_log.keys():
                return len(self.grid_model_list_log[dataset])
            else:
                return 0
        else:
            model_num = 0
            for data in self.grid_model_list_log.keys():
                model_num += len(self.grid_model_list_log[data])
            return model_num

    def convert_vtr_data_to_bin(self, dataset_name, model_name):
        dir_path = os.path.join(self.output_dir, dataset_name)
        if dataset_name in self.dataset_list_log.keys():
            if model_name in self.dataset_list_log[dataset_name]:
                vtr_path = os.path.join(dir_path, model_name) + '.vtr'
                vtr_model = pv.read(vtr_path)
                bin_path = os.path.join(dir_path, model_name) + '.pkl'
                self.save_data(bin_path, vtr_model)
                # 删除vtr 和 g文件
                dir_files_paths = self.get_filelist(dir_path, [])
                for path in dir_files_paths:
                    file_name = os.path.basename(path)
                    name = file_name.split('.')[0]
                    if name == model_name:
                        if not file_name.endswith('.pkl'):
                            os.remove(path)

    def generate_model_by_his(self, model_dir, dataset_name=None, max_num=None):
        # 已生成模型数
        model_saved_num = self.get_model_num(dataset=dataset_name)
        if max_num is not None and max_num > model_saved_num:
            process_model_num = max_num - model_saved_num
        elif max_num is not None:
            process_model_num = 0
        else:
            process_model_num = len(self.model_his_list_log) - model_saved_num
        print('Generating Grid Data')
        save_it = 0
        for dataset in self.dataset_list:
            model_list = self.dataset_list_log[dataset]
            pbar = tqdm(enumerate(model_list), total=len(model_list))
            for it, model_name in pbar:
                if model_name in self.grid_model_list_log[dataset]:
                    continue
                save_it = save_it + 1
                if save_it > process_model_num:
                    break
                file_path = self.model_his_list_log[model_name]
                his_file = os.path.join(self.root, file_path, model_name) + '.his'
                if os.path.exists(his_file):
                    history = os.path.join(his_file)
                    output_dir = os.path.join(model_dir, dataset)
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    output_path = os.path.join(output_dir, model_name)
                    pynoddy.compute_model(history, output_path)
                    # pynoddy.compute_model(history, output_path, sim_type='GEOPHYSICS')
                    nout = pynoddy.output.NoddyOutput(output_path)
                    # nout_geophysics = pynoddy.output.NoddyGeophysics(output_path)
                    # 获取noddy模型参数
                    nx = nout.nx
                    ny = nout.ny
                    nz = nout.nz
                    cell_x = nout.delx
                    cell_y = nout.dely
                    cell_z = nout.delz
                    extent_x = nout.extent_x
                    extent_y = nout.extent_y
                    extent_z = nout.extent_z
                    nout.export_to_vtk()
                    if model_name not in self.model_param_list_log.keys():
                        self.model_param_list_log[model_name] = [nx, ny, nz, cell_x, cell_y, cell_z, extent_x,
                                                                 extent_y, extent_z]
                    if dataset not in self.grid_model_list_log.keys():
                        self.grid_model_list_log[dataset] = []
                    if model_name not in self.grid_model_list_log[dataset]:
                        self.grid_model_list_log[dataset].append(model_name)
                    self.convert_vtr_data_to_bin(dataset, model_name)
        self.save_data(self.grid_model_list_path, self.grid_model_list_log)
        self.save_data(self.model_param_list_path, self.model_param_list_log)

    def extract_his_files_from_tar_files(self, des_dir):
        print('Decompressing and extracting Model Files')
        if self.dataset_list is not None:
            tar_files = self.get_raw_dataset_list()
            for file_name in tar_files:
                # 对 self.dataset_list中所指定的数据集压缩包进行解压
                if file_name in self.dataset_list:  # 加载在数据集列表中的数据集
                    # if file_name in self.dataset_list_log.keys():
                    #     continue
                    file = file_name + '.tar'
                    dir_tmp = os.path.join(self.raw_dir_path, file)
                    tar = tarfile.open(dir_tmp)
                    names = tar.getnames()
                    pbar = tqdm(enumerate(names), total=len(names))
                    for it, name in pbar:
                        if name.endswith('.gz'):
                            # 如果已经生成模型，则不重复解压
                            g_name = name.replace('.gz', '')
                            model_name = os.path.basename(g_name.split('.')[0])
                            if file_name not in self.grid_model_list_log.keys():
                                self.grid_model_list_log[file_name] = []
                            if model_name in self.grid_model_list_log[file_name]:
                                continue
                            if model_name in self.model_his_list_log.keys():
                                continue
                            tar.extract(name, path=des_dir)
                            name = os.path.join(des_dir, name)
                            g_file = gzip.GzipFile(mode="rb", fileobj=open(name, 'rb'))
                            g_name = os.path.join(des_dir, g_name)
                            open(g_name, 'wb').write(g_file.read())
                            g_file.close()
                            _, model_ext = os.path.splitext(g_name)
                            if model_ext == '.his':
                                model_his_dir_path = os.path.dirname(name)
                                if file_name not in self.dataset_list_log.keys():
                                    self.dataset_list_log[file_name] = []
                                self.dataset_list_log[file_name].append(model_name)
                                if model_name not in self.model_his_list_log.keys():  # 记录每个模型文件的地址(相对路径)
                                    self.model_his_list_log[model_name] = os.path.relpath(path=model_his_dir_path,
                                                                                          start=self.root)
                    tar.close()
            self.save_data(self.dataset_list_path, self.dataset_list_log)
            self.save_data(self.model_his_list_path, self.model_his_list_log)

    def delete_gz_files(self):
        print('Deleting GZ Files')
        files = self.get_filelist(self.his_dir_path, [])
        pbar = tqdm(enumerate(files), total=len(files))
        for it, file_path in pbar:
            if file_path.endswith('.gz'):
                os.remove(file_path)

    # file_path: 操作文件夹路径  file_names: list 文件名列表，保留列表内的文件，其余文件删除
    def delete_extra_model_files(self, dir_path, file_names):
        file_paths = self.get_filelist(dir_path, [])
        print('Deleting Extra Vtr Files')
        pbar = tqdm(enumerate(file_paths), total=len(file_paths))
        for it, file_path in pbar:
            file_name = os.path.basename(file_path)
            name = file_name.split('.')[0]
            if name not in file_names:
                os.remove(file_path)

    def delete_extra_his_files(self, dir_path, file_names):
        file_paths = self.get_filelist(dir_path, [])
        print('Deleting Extra His Files')
        pbar = tqdm(enumerate(file_paths), total=len(file_paths))
        for it, file_path in pbar:
            file_name = os.path.basename(file_path)
            name = file_name.split('.')[0]
            if name in file_names:
                os.remove(file_path)

    def delete_dataset_files(self, dataset_name_list: list):
        # 先根据 log 中的记录，删除dataset文件
        # 再清除 log 文件中关于dataset的记录
        for dataset_name in dataset_name_list:
            print('Deleting Dataset:{} Files'.format(dataset_name))
            raw_files = self.get_raw_dataset_list()
            if dataset_name in raw_files:
                raw_file_path = os.path.join(self.raw_dir_path, dataset_name + '.tar')
                if os.path.exists(raw_file_path):
                    os.remove(raw_file_path)
            # 删除 his 文件
            for root, dirs, fs, in os.walk(self.his_dir_path):
                if dataset_name in dirs:
                    his_dir = os.path.join(root, dataset_name)
                    file_names = [file_name.split('.')[0] for file_name in os.listdir(his_dir)]
                    self.delete_extra_his_files(dir_path=his_dir, file_names=file_names)
                    # 删除空目录
                    shutil.rmtree(path=his_dir)
                    break
            # 删除 model_data
            for root, dirs, fs in os.walk(self.output_dir):
                if dataset_name in dirs:
                    model_dir = os.path.join(root, dataset_name)
                    self.delete_extra_model_files(dir_path=model_dir, file_names=os.listdir(model_dir))
                    shutil.rmtree(path=model_dir)
                    break
            # 清理 log 文件中的记录
            if dataset_name in self.dataset_list_log.keys():
                for model_name in self.dataset_list_log[dataset_name]:
                    if model_name in self.model_his_list_log.keys():
                        del self.model_his_list_log[model_name]
                    if model_name in self.grid_model_list_log.keys():
                        del self.grid_model_list_log[model_name]
                    if model_name in self.model_param_list_log.keys():
                        del self.model_param_list_log[model_name]
                del self.dataset_list_log[dataset_name]

    # path: 要传入文件夹绝对路径, filelist:在调用时一定要传[]，因为是对象函数，之前调用产生的路径列表可能会留存
    def get_filelist(self, path, filelist):
        if os.path.exists(path):
            files = os.listdir(path)
        else:
            print('data repo not exist')
            return None
        for file in files:
            cur_path = os.path.join(path, file)
            if os.path.isdir(cur_path):
                self.get_filelist(cur_path, filelist)
            else:
                filelist.append(cur_path)
        return filelist


if __name__ == '__main__':
    noddyData = NoddyModelData(root=r'F:\djc\NoddyDataset', max_model_num=10)
    noddyData.delete_dataset_files(dataset_name_list=['DYKE_TILT_TILT'])
    noddyData.get_grid_model()