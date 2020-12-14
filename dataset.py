import torch.utils.data as data

from PIL import Image
import os, json
import os.path
import numpy as np
from numpy.random import randint
from torchvision import transforms
from glob import glob
import torch


class TSNDataSet(data.Dataset):
    def __init__(self, args, root_path, list_file, new_length=1,
                 image_tmpl='img_{:05d}.jpg', rgb_transform=None,flow_transform=None,
                 random_shift=True, test_mode=False,
                 fix_seed=True,n_aug_support_samples=1):

        self.root_path = root_path
        self.list_file = list_file
        self.new_length = new_length
        self.image_tmpl = image_tmpl
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.n_aug_support_samples = n_aug_support_samples
        self.fix_seed = fix_seed
        self.rgb_transform = rgb_transform
        self.flow_transform = flow_transform
        
        self.num_segments = args.num_segments
        self.modality = args.modality
        self.n_ways = args.train_n
        self.n_shots = args.train_k
        self.n_queries = args.n_queries
        self.n_episodes = args.n_train_runs if test_mode==False else args.n_test_runs

        self.label_mapping = json.load(open('datalists/HMDB51/classes.json'))
        
        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

        self.classes = self.labels
        self.label_index  = self.video_list
        
        

    def _load_image(self, label, vid, idx, modality='RGB'):
        vid_dir = os.path.join(self.root_path, label, vid)
        if modality == 'RGB' or modality == 'RGBDiff':
            img_path = os.path.join(vid_dir, 'img_%05d.jpg' % idx)
            return [Image.open(img_path).convert('RGB')]
        elif modality == 'Flow':
            flow_x_path = os.path.join(vid_dir, 'flow_x_%05d.jpg' % idx)
            flow_y_path = os.path.join(vid_dir, 'flow_y_%05d.jpg' % idx)
            x_img = Image.open(flow_x_path).convert('L')
            y_img = Image.open(flow_y_path).convert('L')
        
            return [x_img, y_img]

    def _parse_list(self):
        self.labels = []
        with open(self.list_file) as f:
            for line in f:
                line = line.strip()
                self.labels.append(line)
        self.video_list = {}
        for label in self.labels:
            vid_dir = os.path.join(self.root_path, label)
            vid_list = os.listdir(vid_dir)
            if len(vid_list) < 1:
                continue
            self.video_list[label] = []
            for vid in vid_list:
                num_frame = len(glob(os.path.join(vid_dir, vid, 'img*.jpg')))
                num_flows = len(glob(os.path.join(vid_dir, vid, 'flow*.jpg')))
                assert  num_flows == (num_frame*2), os.path.join(vid_dir, vid)
                self.video_list[label].append((vid, num_frame))
        
        print('Loaded %d Videos of %d classes' % (sum([len(self.video_list[x]) for x in self.video_list]), len(self.video_list)))
        #self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, num_frames):
        """

        :param record: VideoRecord
        :return: list
        """
        average_duration = (num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif num_frames > self.num_segments:
            offsets = np.sort(randint(num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, num_frames):
        if num_frames > self.num_segments + self.new_length - 1:
            tick = (num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1


    def __getitem__(self, index):
        #if self.fix_seed:
        #    torch.manual_seed(index)
        #cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        cls_sampled = torch.randperm(len(self.classes))[:self.n_ways]
        cls_sampled = [self.classes[x] for x in cls_sampled]
        #print('Sampled Class: ', cls_sampled)
        support_xs_rgb = []
        support_xs_flow = []
        support_ys = []
        query_xs_rgb = []
        query_xs_flow = []
        query_ys = []
        for idx, the_cls in enumerate(cls_sampled):
            vids = self.label_index[the_cls]
            #support_xs_ids_sampled = np.random.choice(range(len(vids)), self.n_shots, False)
            support_xs_ids_sampled = torch.randperm(len(vids))[:self.n_shots]
            for idx_s in support_xs_ids_sampled:
                vid, num_frames = self.label_index[the_cls][idx_s]
                segment_indices = self._sample_indices(num_frames) if self.random_shift else self._get_test_indices(num_frames)
                if self.modality in ['RGB', 'Joint']:
                    supp_rgb, supp_label = self.get(the_cls, vid, num_frames, segment_indices, 'RGB')
                    support_xs_rgb.append(supp_rgb)
                if self.modality in ['Flow', 'Joint']:
                    supp_flow, supp_label = self.get(the_cls, vid, num_frames, segment_indices, 'Flow')
                    support_xs_flow.append(supp_flow)
                support_ys.append(supp_label)

            query_xs_ids = np.setxor1d(np.arange(len(vids)), support_xs_ids_sampled)
            
            #query_xs_ids_sampled = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs_ids_sampled = torch.randperm(len(query_xs_ids))[:self.n_queries]
            for idx_s in query_xs_ids_sampled:
                #record = self.label_index[the_cls][index]
                vid, num_frames = self.label_index[the_cls][idx_s]
                segment_indices = self._sample_indices(num_frames) if self.random_shift else self._get_test_indices(num_frames)
                if self.modality in ['RGB', 'Joint']:
                    query_rgb, query_label = self.get(the_cls, vid, num_frames, segment_indices, 'RGB')
                    query_xs_rgb.append(query_rgb)
                if self.modality in ['Flow', 'Joint']:
                    query_flow, query_label = self.get(the_cls, vid, num_frames, segment_indices, 'Flow')
                    query_xs_flow.append(query_flow)
                query_ys.append(query_label)

        if len(support_xs_rgb) > 0:
            support_xs_rgb = torch.stack(support_xs_rgb, dim=0)
        if len(support_xs_flow) > 0:
            support_xs_flow = torch.stack(support_xs_flow, dim=0)
        if len(query_xs_rgb) > 0:
            query_xs_rgb = torch.stack(query_xs_rgb, dim=0)
        if len(query_xs_flow) > 0:
            query_xs_flow = torch.stack(query_xs_flow, dim=0)

        support_ys = torch.from_numpy(np.array(support_ys))
        query_ys = torch.from_numpy(np.array(query_ys))

        return support_xs_rgb, support_xs_flow, support_ys, query_xs_rgb, query_xs_flow, query_ys

    def get(self, label, vid, num_frames, indices, modality='RGB'):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(label, vid, p, modality)
                images.extend(seg_imgs)
                if p < num_frames:
                    p += 1
        
        if modality == 'RGB':
            process_data = self.rgb_transform(images)
        elif modality == 'Flow':
            process_data = self.flow_transform(images)
        
        label = self.label_mapping[label]
        return process_data, label

    def __len__(self):
        return self.n_episodes
