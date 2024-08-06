def set_map_points_data_labels_to_base_grid(self, points_data: PointSet, sid=-1):
    if self.map_flag:
        cells_series = self._base_grid_labels
    else:
        cells_series = np.full((len(self._base_grid_points),), fill_value=self.default_value)
    # 如果设置了验证钻孔比例，则
    self.update_train_val_split_state(sid=sid)
    log_id = len(self.sample_operator) - 1
    if log_id in self.geo_sample_data_val_map.keys():
        val_points_data = points_data.get_points_data_by_ids(ids=self.geo_sample_data_val_map[log_id]['val'])
        train_points_data = points_data.get_points_data_by_ids(ids=self.geo_sample_data_val_map[log_id]['train'])
        test_points_data = points_data.get_points_data_by_ids(ids=self.geo_sample_data_val_map[log_id]['test'])
        val_cell_indices, val_cell_labels = self.map_base_grid_points_by_sample_data(sample_data=val_points_data)
        train_cell_indices, train_cell_labels = self.map_base_grid_points_by_sample_data(
            sample_data=train_points_data)
        test_cell_indices, test_cell_labels = self.map_base_grid_points_by_sample_data(sample_data=test_points_data)
        self.train_indexes.append(train_cell_indices)
        self.val_indexes.append(val_cell_indices)
        self.test_indexes.append(test_cell_indices)
        # 合并
        cell_indices = np.hstack((val_cell_indices, train_cell_indices, test_cell_indices))
        cell_labels = np.hstack((val_cell_labels, train_cell_labels, test_cell_labels))
        cell_indices, cell_labels = remove_repeated_elements_with_lists(
            list_item_1=cell_indices, list_item_2=cell_labels)
    else:
        cell_indices, cell_labels = self.map_base_grid_points_by_sample_data(sample_data=points_data)
        self.train_indexes.append(cell_indices)
    cells_series[cell_indices] = cell_labels
    self.map_flag = True
    self._base_grid.grid_points_series = cells_series
    self._base_grid_labels = cells_series
    self._base_grid.vtk_data.cell_data['Scalar Field'] = cells_series
    self._base_grid.classes = np.unique(cells_series)
    self._base_grid.classes_num = len(self._base_grid.classes)
    if self.default_value in np.unique(cells_series):
        self._base_grid.classes_num -= 1