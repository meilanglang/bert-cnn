#!/usr/bin/env python
import numpy as np
def load_data():
    data=np.load('cellline.npz')
    emb_ep=(data['enhancer_pos'])
    emb_en=(data['enhancer_neg'])
    emb_pp=(data['promoter_pos'])
    emb_pn=(data['promoter_neg'])
    train_emb_e = np.concatenate([emb_ep, emb_en])
    train_emb_p = np.concatenate([emb_pp, emb_pn])
    label = np.concatenate([np.ones([len(emb_ep),1]), np.zeros([len(emb_en),1])])
    np.savez(celline+'_bagging_train.npz',enhancer=train_emb_e,promoter=train_emb_p,label=label)

load_data()
