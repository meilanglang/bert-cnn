#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/env python
import numpy as np
#from sample import load_data

def load_data():
    data=np.load('cellline_test.npz')
    emb_ep=(data['enhancer_pos'])
    emb_en=(data['enhancer_neg'])
    emb_pp=(data['promoter_pos'])
    emb_pn=(data['promoter_neg'])
    #label=(data['label'])
    test_emb_e = np.concatenate([emb_ep, emb_en])
    test_emb_p = np.concatenate([emb_pp, emb_pn])
    label = np.concatenate([np.ones([len(emb_ep),1]), np.zeros([len(emb_en),1])])
    print(test_emb_e.shape,test_emb_p.shape,label.shape)
    return test_emb_e,test_emb_p,label

def load_data_one_hot():
    data=np.load('cellline_hot_test.npz')
    emb_ep=(data['enhancer_pos'])
    emb_en=(data['enhancer_neg'])
    emb_pp=(data['promoter_pos'])
    emb_pn=(data['promoter_neg'])
    #label=(data['label'])
    test_emb_e = np.concatenate([emb_ep, emb_en])
    test_emb_p = np.concatenate([emb_pp, emb_pn])
    label = np.concatenate([np.ones([len(emb_ep),1]), np.zeros([len(emb_en),1])])
    print(test_emb_e.shape,test_emb_p.shape,label.shape)
    return test_emb_e,test_emb_p,label
