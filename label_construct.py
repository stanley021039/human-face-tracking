import re

ori_path = 'label/wider_face_split/wider_face_val_bbx_gt.txt'
save_path = 'dataset/valid_label/'
with open(ori_path, 'r') as f:
    new_txt_path = ''
    while True:
        L = f.readline()
        if 'j' in L:
            # new txt named L
            print(L)
            L = L.split('/')[1]
            new_txt_path = save_path + L.replace('jpg\n', 'txt')
            wf = open(new_txt_path, 'w')
            f.readline()
        else:
            wf.write(L)

