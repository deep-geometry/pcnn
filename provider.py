""" Based on point net data
downladed from: https://github.com/charlesq34/pointnet
"""
import os
import sys
import numpy as np
import scipy as sp
import scipy.spatial
import h5py
import point_sampling_utils as psu
from abc import ABCMeta,abstractmethod


class Provider:

    def getDataFiles(self,file):
        pass

    def shuffle_data(self,data, labels):
        """ Shuffle data and labels.
            Input:
              data: B,N,... numpy array
              label: B,... numpy array
            Return:
              shuffled data, label and shuffle indices
        """
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        return data[idx, ...], labels[idx], idx

    def translate_point_cloud(self,batch_data):

        translated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
            xyz2 = np.random.uniform(low=-0.2,high=0.2,size=[3])

            shape_pc = batch_data[k, ...]

            translated_data[k, ...] = np.add(np.multiply(shape_pc, xyz1), xyz2)
        return translated_data

    def rotate_point_cloud_by_angle(self,batch_data, rotation_angle):
        """ Rotate the point cloud along up direction with certain angle.
          Input:
            BxNx3 array, original batch of point clouds
          Return:
            BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            # rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    def rotate_point_cloud(self,batch_data):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    def jitter_point_cloud(self,batch_data, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """
        B, N, C = batch_data.shape
        assert (clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
        jittered_data += batch_data
        return jittered_data


class ClassificationProvider(Provider):
    def __init__(self,download = True):
        self.BASE_DIR = '.'
        sys.path.append(self.BASE_DIR)
        DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        if download and not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        if download and  not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
            www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
            zipfile = os.path.basename(www)
            os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
            os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
            os.system('rm %s' % (zipfile))

        self.train_files = os.path.join(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048'),'train_files.txt')
        self.test_files = os.path.join(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048'),'test_files.txt')

    def getTestDataFiles(self):
        return self.getDataFiles(self.test_files)

    def getTrainDataFiles(self):
        return self.getDataFiles(self.train_files)

    def getDataFiles(self,list_filename):
        return [line.rstrip() for line in open(list_filename)]

    def load_h5(self,h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:]
        return (data, label)

    def loadDataFile(self,filename):
        return self.load_h5(os.path.join(self.BASE_DIR,filename))

    def read_off(file):
        if 'OFF' != file.readline().strip():
            raise ('Not a valid OFF header')
        n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = []
        for i_vert in range(n_verts):
            verts.append([float(s) for s in file.readline().strip().split(' ')])
        faces = []
        for i_face in range(n_faces):
            faces.append([int(s) for s in file.readline().strip().split(' ')][1:])
        return verts, faces

    def load_h5_data_label_seg(h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:]
        seg = f['pid'][:]
        return (data, label, seg)


class SebastianProvider(object):
    def __init__(self, traindir, testdir, batch_size, points_per_patch, normalize_data=True):
        self.traindir = traindir
        self.testdir = testdir

        self.train_file_list = list(os.listdir(traindir))
        self.num_train_patches = len(self.train_file_list)
        print(self.num_train_patches)

        self.test_file_list = list(os.listdir(testdir))
        self.num_test_patches = len(self.test_file_list)

        self.points_per_patch = points_per_patch
        self.batch_size = batch_size

        self.access_indices = np.random.permutation(self.num_train_patches)
        self.normalize_data = normalize_data

        self.train_points, self.train_normals = self.gen_data(self.traindir)
        self.test_points, self.test_normals = self.gen_data(self.testdir)

    def gen_data(self, directory):
        file_list = list(os.listdir(directory))
        num_patches = len(file_list)
        points = np.zeros([num_patches, self.points_per_patch, 3], dtype=np.float32)
        normals = np.zeros([num_patches, 3], np.float32)
        print("Generating %d patches" % num_patches)
        for shape_ind, shape_name in enumerate(file_list):
            # print('getting information for shape %s' % shape_name)

            # load from text file and save in more efficient numpy format
            point_filename = os.path.join(directory, shape_name)
            v, _, n = psu.read_obj(point_filename)
            pts = v

            scale_factor = 1.0
            min_translate_factor = np.zeros(3)

            if self.normalize_data:
                min_translate_factor = np.min(pts, axis=0)
                pts -= min_translate_factor
                scale_factor = 1.0 / np.max(pts)
                pts *= scale_factor

            kdtree = sp.spatial.KDTree(pts)
            centroid = np.mean(pts, axis=0)
            _, i = kdtree.query(centroid)
            center_vertex = pts[i, :].astype(np.float32)
            pts -= center_vertex

            points[shape_ind, :, :] = pts.astype(np.float32)
            normals[shape_ind, :] = n[i, :].astype(np.float32)

            # shape = {'p': pts.astype(np.float32), 'n': n.astype(np.float32),
            #          'ctr': center_vertex, 'ctr_i': i,
            #          'scale_factor': scale_factor,
            #          'min_translate_factor': min_translate_factor }
            # self.shapes.append(shape)
        return points, normals

    def reshuffle(self):
        self.access_indices = np.random.permutation(self.num_train_patches)

    @property
    def num_batches(self):
        return int(np.ceil(self.num_train_patches / self.batch_size))

    def getTrainDataFiles(self):
        return ["dummy"]

    def getTestDataFiles(self):
        return ["dummy"]

    def loadDataFile(self, is_train=True):
        if is_train:
            return self.train_points[self.access_indices, :, :], \
                   self.train_normals[self.access_indices, :]
        else:
            return self.test_points, self.test_normals

    def shuffle_data(self, data, labels):
        """ Shuffle data and labels.
            Input:
              data: B,N,... numpy array
              label: B,... numpy array
            Return:
              shuffled data, label and shuffle indices
        """
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        return data[idx, ...], labels[idx], idx