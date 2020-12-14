import os

train_file = 'datalists/kinetics-100/train.list'
val_file = 'datalists/kinetics-100/val.list'
test_file = 'datalists/kinetics-100/test.list'

root_dir = '/dvmm-filer2/datasets/Kinetics/400/raw_video/kinetics_val/videos_trimmed'
root_dir_supp = '/dvmm-filer2/datasets/Kinetics/400/raw_video/kinetics_supp/valid'

assert os.path.exists(root_dir)
assert os.path.exists(root_dir_supp)

k=0
with open(val_file) as f:
    for line in f:
        line = line.strip().split('/')
        label = line[0]
        url = line[1]
        label = label.replace(' ', '_')
        start = url[12:18]
        url = url[:11]
        start = int(start)
        url = '_'.join([url, str(start), '10'])
        #print(label, url)

        vid_file = os.path.join(root_dir, label, url+'.mp4')
        if not os.path.exists(vid_file):
            vid_file = os.path.join(root_dir_supp, label, url+'.mp4')
            if os.path.exists(vid_file):
                print(url + ' in supp')
            else:
                k +=1
                print(vid_file)

print(k)
