import xgboost as xgb
import argparse
#from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
#import mplhep as hep
import tqdm
import awkward as ak
import dask_awkward as dak
import copy
import time
from distributed import LocalCluster, Client, progress
import os
import copy

def customROC_curve_AN(label, pred, weight):
    """
    generates signal and background efficiency consistent with the AN,
    as described by Fig 4.6 of Dmitry's PhD thesis
    """
    # we assume sigmoid output with labels 0 = background, 1 = signal
    thresholds = np.linspace(start=0,stop=1, num=500) 
    effBkg_total = -99*np.ones_like(thresholds) # effBkg = false positive rate
    effSig_total = -99*np.ones_like(thresholds) # effSig = true positive rate
    for ix in range(len(thresholds)):
        threshold = thresholds[ix]
        # get FP and TP
        positive_filter = (pred > threshold)
        falsePositive_filter = positive_filter & (label == 0)
        FP = np.sum(weight[falsePositive_filter])#  FP = false positive
        truePositive_filter = positive_filter & (label == 1)
        TP = np.sum(weight[truePositive_filter])#  TP = true positive
        

        # get TN and FN
        negative_filter = (pred <= threshold) # just picked negative to be <=
        trueNegative_filter = negative_filter & (label == 0)
        TN = np.sum(weight[trueNegative_filter])#  TN = true negative
        falseNegative_filter = negative_filter & (label == 1)
        FN = np.sum(weight[falseNegative_filter])#  FN = false negative

        


        # effBkg = TN / (TN + FP) # Dmitry PhD thesis definition
        # effSig = FN / (FN + TP) # Dmitry PhD thesis definition
        effBkg = FP / (TN + FP) # AN-19-124 ggH Cat definition
        effSig = TP / (FN + TP) # AN-19-124 ggH Cat definition
        effBkg_total[ix] = effBkg
        effSig_total[ix] = effSig

        # print(f"ix: {ix}") 
        # print(f"threshold: {threshold}")
        # print(f"effBkg: {effBkg}")
        # print(f"effSig: {effSig}")
        
        
        # sanity check
        assert ((np.sum(positive_filter) + np.sum(negative_filter)) == len(pred))
        total_yield = FP + TP + FN + TN
        assert(np.isclose(total_yield, np.sum(weight)))
        # print(f"total_yield: {total_yield}")
        # print(f"np.sum(weight): {np.sum(weight)}")
    
    # print(f"np.sum(effBkg_total ==-99) : {np.sum(effBkg_total ==-99)}")
    # print(f"np.sum(effSig_total ==-99) : {np.sum(effSig_total ==-99)}")
    # neither_zeroNorOne = ~((label == 0) | (label == 1))
    # print(f"np.sum(neither_zeroNorOne) : {np.sum(neither_zeroNorOne)}")
    effBkg_total[np.isnan(effBkg_total)] = 1
    effSig_total[np.isnan(effSig_total)] = 1
    # print(f"effBkg_total: {effBkg_total}")
    # print(f"effSig_total: {effSig_total}")
    # print(f"thresholds: {thresholds}")
    # raise ValueError
    return (effBkg_total, effSig_total, thresholds)

#training_features = ['dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_ebe_mass_res', 'dimuon_ebe_mass_res_rel', 'dimuon_eta', 'dimuon_mass', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 'dimuon_pt_log', 'jet1_eta nominal', 'jet1_phi nominal', 'jet1_pt nominal', 'jet1_qgl nominal', 'jet2_eta nominal', 'jet2_phi nominal', 'jet2_pt nominal', 'jet2_qgl nominal', 'jj_dEta nominal', 'jj_dPhi nominal', 'jj_eta nominal', 'jj_mass nominal', 'jj_mass_log nominal', 'jj_phi nominal', 'jj_pt nominal', 'll_zstar_log nominal', 'mmj1_dEta nominal', 'mmj1_dPhi nominal', 'mmj2_dEta nominal', 'mmj2_dPhi nominal', 'mmj_min_dEta nominal', 'mmj_min_dPhi nominal', 'mmjj_eta nominal', 'mmjj_mass nominal', 'mmjj_phi nominal', 'mmjj_pt nominal', 'mu1_eta', 'mu1_iso', 'mu1_phi', 'mu1_pt', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt', 'mu2_pt_over_mass', 'zeppenfeld nominal']
#training_features = ['dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_eta', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 'dimuon_pt_log', 'jet1_eta_nominal', 'jet1_phi_nominal', 'jet1_pt_nominal', 'jet1_qgl_nominal', 'jet2_eta_nominal', 'jet2_phi_nominal', 'jet2_pt_nominal', 'jet2_qgl_nominal', 'jj_dEta_nominal', 'jj_dPhi_nominal', 'jj_eta_nominal', 'jj_mass_nominal', 'jj_mass_log_nominal', 'jj_phi_nominal', 'jj_pt_nominal', 'll_zstar_log_nominal', 'mmj1_dEta_nominal', 'mmj1_dPhi_nominal', 'mmj2_dEta_nominal', 'mmj2_dPhi_nominal', 'mmj_min_dEta_nominal', 'mmj_min_dPhi_nominal', 'mmjj_eta_nominal', 'mmjj_mass_nominal', 'mmjj_phi_nominal', 'mmjj_pt_nominal', 'mu1_eta', 'mu1_iso', 'mu1_phi', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt_over_mass', 'zeppenfeld_nominal']

training_features = ['dimuon_cos_theta_cs_pisa', 'dimuon_eta', 'dimuon_phi_cs_pisa', 'dimuon_pt', 'jet1_eta_nominal', 'jet1_pt_nominal', 'jet2_eta_nominal', 'jet2_pt_nominal', 'jj_dEta_nominal', 'jj_dPhi_nominal', 'jj_mass_nominal', 'mmj1_dEta_nominal', 'mmj1_dPhi_nominal',  'mmj_min_dEta_nominal', 'mmj_min_dPhi_nominal', 'mu1_eta', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_pt_over_mass', 'zeppenfeld_nominal', 'njets_nominal'] # AN 19-124

#training_features = ['dimuon_cos_theta_cs', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_eta', 'dimuon_phi', 'dimuon_phi_cs', 'dimuon_pt', 'dimuon_pt_log', 'jet1_eta_nominal', 'jet1_phi_nominal', 'jet1_pt_nominal', 'jet2_eta_nominal', 'jet2_phi_nominal', 'jet2_pt_nominal',  'jj_dEta_nominal', 'jj_dPhi_nominal', 'jj_eta_nominal', 'jj_mass_nominal', 'jj_mass_log_nominal', 'jj_phi_nominal', 'jj_pt_nominal', 'll_zstar_log_nominal', 'mmj1_dEta_nominal', 'mmj1_dPhi_nominal', 'mmj2_dEta_nominal', 'mmj2_dPhi_nominal', 'mmj_min_dEta_nominal', 'mmj_min_dPhi_nominal', 'mmjj_eta_nominal', 'mmjj_mass_nominal', 'mmjj_phi_nominal', 'mmjj_pt_nominal', 'mu1_eta', 'mu1_iso', 'mu1_phi', 'mu1_pt_over_mass', 'mu2_eta', 'mu2_iso', 'mu2_phi', 'mu2_pt_over_mass', 'zeppenfeld_nominal']

training_samples = {
        # "background": ["dy_M-100To200"],
        # "signal": ["ggh_powheg"],
        "background": ["dy_m105_160_amc"],
        "signal": ["ggh_amcPS", "vbf_powheg_dipole"],
        
        #"ignore": [
        #    "tttj",
        #    "tttt",
        #    "tttw",
        #    "ttwj",
        #    "ttww",
        #   "ttz",
        #    "st_s",
        #    "st_t_antitop",
        #    "st_tw_top",
        #    "st_tw_antitop",
        #    "zz_2l2q",
        #],
    }

def prepare_dataset(df, ds_dict):
    # Convert dictionary of datasets to a more useful dataframe
    df_info = pd.DataFrame()
    train_samples = []
    for icls, (cls, ds_list) in enumerate(ds_dict.items()):
        for ds in ds_list:
            df_info.loc[ds, "dataset"] = ds
            df_info.loc[ds, "iclass"] = -1
            if cls != "ignore":
                train_samples.append(ds)
                df_info.loc[ds, "class_name"] = cls
                df_info.loc[ds, "iclass"] = icls
    df_info["iclass"] = df_info["iclass"].fillna(-1).astype(int)
    df = df[df.dataset.isin(df_info.dataset.unique())]
    
    # Assign numerical classes to each event
    cls_map = dict(df_info[["dataset", "iclass"]].values)
    cls_name_map = dict(df_info[["dataset", "class_name"]].values)
    df["class"] = df.dataset.map(cls_map)
    df["class_name"] = df.dataset.map(cls_name_map)
    df.loc[:,'mu1_pt_over_mass'] = np.divide(df['mu1_pt'], df['dimuon_mass'])
    df.loc[:,'mu2_pt_over_mass'] = np.divide(df['mu2_pt'], df['dimuon_mass'])
    df[df['njets_nominal']<2]['jj_dPhi_nominal'] = -1
    #df[training_features].fillna(0, inplace=True)
    # initialze the training wgts
    df["wgt_nominal_orig"] = copy.deepcopy(df["wgt_nominal"])
    df[df['dataset']=="ggh_amcPS"].loc[:,'wgt_nominal'] = np.divide(df[df['dataset']=="ggh_amcPS"]['wgt_nominal'], df[df['dataset']=="ggh_amcPS"]['dimuon_ebe_mass_res'])
    # df.loc[df['dataset']=="ggh_powheg",'wgt_nominal'] = np.divide(df[df['dataset']=="ggh_powheg"]['wgt_nominal'], df[df['dataset']=="ggh_powheg"]['dimuon_ebe_mass_res'])
    #df[training_features].fillna(-99,inplace=True)
    df.fillna(-99,inplace=True)
    #print(df.head)
    # columns_print = ['njets_nominal','jj_dPhi_nominal','jj_mass_log_nominal', 'jj_phi_nominal', 'jj_pt_nominal', 'll_zstar_log_nominal', 'mmj1_dEta_nominal',]
    # columns_print = ['njets_nominal','jj_dPhi_nominal','jj_mass_log_nominal', 'jj_phi_nominal', 'jj_pt_nominal', 'll_zstar_log_nominal', 'mmj1_dEta_nominal','jet2_pt_nominal']
    # columns2 = ['mmj1_dEta_nominal', 'mmj1_dPhi_nominal', 'mmj2_dEta_nominal', 'mmj2_dPhi_nominal', 'mmj_min_dEta_nominal', 'mmj_min_dPhi_nominal']
    # with open("df.txt", "w") as f:
    #     print(df[columns_print], file=f)
    # with open("df2.txt", "w") as f:
    #     print(df[columns2], file=f)
    # print(df[df['dataset']=="ggh_powheg"].head)
    return df

def classifier_train(df, args):
    if args['dnn']:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, Activation, Input, Dropout, Concatenate, Lambda, BatchNormalization
        from tensorflow.keras import backend as K
    if args['bdt']:
        import xgboost as xgb
        from xgboost import XGBClassifier
        import pickle
    def scale_data(inputs, label):
        x_mean = np.mean(x_train[inputs].values,axis=0)
        x_std = np.std(x_train[inputs].values,axis=0)
        training_data = (x_train[inputs]-x_mean)/x_std
        validation_data = (x_val[inputs]-x_mean)/x_std
        output_path = args["output_path"]
        base_path = f"{output_path}/{name}_{year}"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        np.save(f"{base_path}/scalers_{name}_{year}_{label}", [x_mean, x_std])
        return training_data, validation_data
    
    def scale_data_withweight(inputs, label):
        masked_x_train = np.ma.masked_array(x_train[x_train[inputs]!=-99][inputs], np.isnan(x_train[x_train[inputs]!=-99][inputs]))
        #x_mean = np.average(x_train[inputs].values,axis=0, weights=df_train['training_wgt'].values)
        x_mean = np.average(masked_x_train,axis=0, weights=df_train['training_wgt'].values).filled(np.nan)
        #x_std = np.std(x_train[inputs].values,axis=0)
        #masked_x_std = np.ma.masked_array(x_train[x_train[inputs]!=-99][inputs], np.isnan(x_train[x_train[inputs]!=-99][inputs]))
        #x_std = np.average((x_train[inputs].values-x_mean)**2,axis=0, weights=df_train['training_wgt'].values)
        x_std = np.average((masked_x_train-x_mean)**2,axis=0, weights=df_train['training_wgt'].values).filled(np.nan)
        sumw2 = (df_train['training_wgt']**2).sum()
        sumw = df_train['training_wgt'].sum()
        
        x_std = np.sqrt(x_std/(1-sumw2/sumw**2))
        training_data = (x_train[inputs]-x_mean)/x_std
        validation_data = (x_val[inputs]-x_mean)/x_std
        evaluation_data = (x_eval[inputs]-x_mean)/x_std
        output_path = args["output_path"]
        # print(output_path)
        # print(name)
        base_path = f"{output_path}/{name}_{year}"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        np.save(f'{base_path}/scalers_{name}_{label}', [x_mean, x_std]) #label contains year
        return training_data, validation_data, evaluation_data

    nfolds = 4
    classes = df.dataset.unique()
    #print(df["class"])
    #cls_idx_map = {dataset:idx for idx,dataset in enumerate(classes)}
    add_year = (args['year']=='')
    #df = prepare_features(df, args, add_year)
    #df['cls_idx'] = df['dataset'].map(cls_idx_map)
    print("Training features: ", training_features)
    for i in range(nfolds):
        if args['year']=='':
            label = f"allyears_{args['label']}_{i}"
        else:
            label = f"{args['year']}_{args['label']}{i}"
        
        train_folds = [(i+f)%nfolds for f in [0,1]]
        val_folds = [(i+f)%nfolds for f in [2]]
        eval_folds = [(i+f)%nfolds for f in [3]]

        print(f"Train classifier #{i+1} out of {nfolds}")
        print(f"Training folds: {train_folds}")
        print(f"Validation folds: {val_folds}")
        print(f"Evaluation folds: {eval_folds}")
        print(f"Samples used: ",df.dataset.unique())
        
        train_filter = df.event.mod(nfolds).isin(train_folds)
        val_filter = df.event.mod(nfolds).isin(val_folds)
        eval_filter = df.event.mod(nfolds).isin(eval_folds)
        
        other_columns = ['event']
        
        df_train = df[train_filter]
        df_val = df[val_filter]
        df_eval = df[eval_filter]
        
        x_train = df_train[training_features]
        #y_train = df_train['cls_idx']
        y_train = df_train['class']
        x_val = df_val[training_features]
        x_eval = df_eval[training_features]
        #y_val = df_val['cls_idx']
        y_val = df_val['class']
        y_eval = df_eval['class']

        #df_train['cls_avg_wgt'] = 1.0
        #df_val['cls_avg_wgt'] = 1.0
        
        for icls, cls in enumerate(classes):
            train_evts = len(y_train[y_train==icls])
            df_train.loc[y_train==icls,'cls_avg_wgt'] = df_train.loc[y_train==icls,'wgt_nominal'].values.mean()
            df_val.loc[y_val==icls,'cls_avg_wgt'] = df_val.loc[y_val==icls,'wgt_nominal'].values.mean()
            df_eval.loc[y_eval==icls,'cls_avg_wgt'] = df_eval.loc[y_eval==icls,'wgt_nominal'].values.mean()
            print(f"{train_evts} training events in class {cls}")
        
        df_train['training_wgt'] = df_train['wgt_nominal']/df_train['cls_avg_wgt']
        df_val['training_wgt'] = df_val['wgt_nominal']/df_val['cls_avg_wgt']
        df_eval['training_wgt'] = df_eval['wgt_nominal']/df_eval['cls_avg_wgt']
        
        # scale data
        #x_train, x_val = scale_data(training_features, label)#Last used
        x_train, x_val, x_eval = scale_data_withweight(training_features, label)
        x_train[other_columns] = df_train[other_columns]
        x_val[other_columns] = df_val[other_columns]
        x_eval[other_columns] = df_eval[other_columns]

        # load model
        if args['dnn']:
            input_dim = len(training_features)
            inputs = Input(shape=(input_dim,), name = label+'_input')
            x = Dense(100, name = label+'_layer_1', activation='tanh')(inputs)
            x = Dropout(0.2)(x)
            x = BatchNormalization()(x)
            x = Dense(100, name = label+'_layer_2', activation='tanh')(x)
            x = Dropout(0.2)(x)
            x = BatchNormalization()(x)
            x = Dense(100, name = label+'_layer_3', activation='tanh')(x)
            x = Dropout(0.2)(x)
            x = BatchNormalization()(x)
            outputs = Dense(1, name = label+'_output',  activation='sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
            model.summary()

            #history = model.fit(x_train[training_features], y_train, epochs=100, batch_size=1024,\
            #                    sample_weight=df_train['training_wgt'].values, verbose=1,\
            #                    validation_data=(x_val[training_features], y_val, df_val['training_wgt'].values), shuffle=True)
            history = model.fit(x_train[training_features], y_train, epochs=10, batch_size=1024,
                                verbose=1,
                                validation_data=(x_val[training_features], y_val), shuffle=True)
            
            util.save(history.history, f"output/trained_models/history_{label}_dnn.coffea")
            y_pred = model.predict(x_val).ravel()
            nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(y_val, y_pred)
            auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
            #plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='Neural Network (auc = %0.3f)' % auc_keras)
            roc_auc_gus = auc(nn_fpr_keras,nn_tpr_keras)
            fig, ax = plt.subplots(1,1)
            ax.plot(nn_fpr_keras, nn_tpr_keras, label='Raw ROC curve (area = %0.2f)' % roc_auc)
            #ax.plot(fpr_gus, tpr_gus, label='Gaussian ROC curve (area = %0.2f)' % roc_auc_gus)
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver operating characteristic example')
            ax.legend(loc="lower right")
            fig.savefig(f"output/trained_models/test_{label}.png")
            model.save(f"output/trained_models/test_{label}.h5")        
        if args['bdt']:
            seed = 7
            xp_train = x_train[training_features].values
            xp_val = x_val[training_features].values
            xp_eval = x_eval[training_features].values
            y_train = y_train.values
            y_val = y_val.values
            y_eval = y_eval.values
            w_train = df_train['training_wgt'].values
            w_val = df_val['training_wgt'].values
            w_eval = df_eval['training_wgt'].values

            weight_nom_train = df_train['wgt_nominal_orig'].values
            weight_nom_val = df_val['wgt_nominal_orig'].values
            weight_nom_eval = df_eval['wgt_nominal_orig'].values
            
            shuf_ind_tr = np.arange(len(xp_train))
            np.random.shuffle(shuf_ind_tr)
            shuf_ind_val = np.arange(len(xp_val))
            np.random.shuffle(shuf_ind_val)
            shuf_ind_eval = np.arange(len(xp_eval))
            np.random.shuffle(shuf_ind_eval)
            xp_train = xp_train[shuf_ind_tr]
            xp_val = xp_val[shuf_ind_val]
            y_train = y_train[shuf_ind_tr]
            y_val = y_val[shuf_ind_val]
            #print(np.isnan(xp_train).any())
            #print(np.isnan(y_train).any())
            #print(np.isinf(xp_train).any())
            #print(np.isinf(y_train).any())
            #print(np.isfinite(x_train).all())
            #print(np.isfinite(y_train).all())
            
            w_train = w_train[shuf_ind_tr]
            w_val = w_val[shuf_ind_val]

            xp_eval = xp_eval[shuf_ind_eval]
            y_eval = y_eval[shuf_ind_eval]

            weight_nom_train = weight_nom_train[shuf_ind_tr]
            weight_nom_val = weight_nom_val[shuf_ind_val]
            weight_nom_eval = weight_nom_eval[shuf_ind_eval]
            #data_dmatrix = xgb.DMatrix(data=X,label=y)
            """
            model = xgb.XGBClassifier(max_depth=7,#for 2018
                                  #max_depth=6,previous value
                                  n_estimators=10000,
                                  #n_estimators=100,
                                  early_stopping_rounds=50, eval_metric="logloss",
                                  #learning_rate=0.001,#for 2018
                                  learning_rate=0.0034,#previous value
                                  reg_alpha=0.680159426755822,
                                  colsample_bytree=0.47892268305051233,
                                  min_child_weight=20,
                                  subsample=0.5606,
                                  reg_lambda=16.6,
                                  gamma=24.505,
                                  n_jobs=35,
                                      tree_method='gpu_hist')
                                  #tree_method='hist')
            """                   
            model = xgb.XGBClassifier(max_depth=4,#for 2018
                                      #max_depth=6,previous value
                                      n_estimators=100000,
                                      #n_estimators=100,
                                      early_stopping_rounds=3, #80
                                      eval_metric="logloss",
                                      #learning_rate=0.001,#for 2018
                                      learning_rate=0.1,#previous value
                                      #reg_alpha=0.680159426755822,
                                      #colsample_bytree=0.47892268305051233,
                                      colsample_bytree=0.5,
                                      min_child_weight=3,
                                      subsample=0.5,
                                      #reg_lambda=16.6,
                                      #gamma=24.505,
                                      #n_jobs=35,
                                      tree_method='hist')
                                      #tree_method='hist')
            """
            #Multiclass
            model = XGBClassifier(max_depth=7,#for 2018
                                  #max_depth=6,previous value
                                  n_estimators=100000,
                                  #n_estimators=100,
                                  objective='multi:softmax',
                                  num_class = classes,
                                  #learning_rate=0.001,#for 2018
                                  learning_rate=0.0034,#previous value
                                  reg_alpha=0.680159426755822,
                                  colsample_bytree=0.47892268305051233,
                                  min_child_weight=20,
                                  subsample=0.5606,
                                  reg_lambda=16.6,
                                  gamma=24.505,
                                  n_jobs=5,
                                  tree_method='gpu_hist')

            
            model = XGBClassifier(max_depth=7,
                                  n_estimators=10000,
                                  #n_estimators=100,
                                  learning_rate=0.007972,
                                  reg_alpha=23.81,
                                  colsample_bytree=0.6506,
                                  min_child_weight=7.109,
                                  subsample=0.7306,
                                  reg_lambda=0.1708,
                                  gamma=15.77,
                                  n_jobs=1,
                                  tree_method='hist')
            """
            print(model)
            #eval_set = [(x_train[training_features], y_train), (x_val[training_features], y_val)]
            #eval_set = [(x_train[training_features], y_train, df_train['training_wgt'].values), (x_val[training_features], y_val, df_val['training_wgt'].values)]#Last used
            #eval_set = [(xp_train, y_train, df_train['training_wgt'].values), (xp_val, y_val, df_val['training_wgt'].values)]#Last used
            #eval_set = [(xp_train, y_train, w_train), (xp_val, y_val, w_val)]#Last used
            eval_set = [(xp_train, y_train), (xp_val, y_val)]#Last used
            #model.fit(x_train[training_features].values, y_train, sample_weight = df_train['training_wgt'].values, early_stopping_rounds=50, eval_metric="logloss", eval_set=eval_set, verbose=True)#Last used
            model.fit(xp_train, y_train, sample_weight = w_train, eval_set=eval_set, verbose=False)


            y_pred_signal_val = model.predict_proba(xp_val)[:, 1].ravel()
            y_pred_signal_train = model.predict_proba(xp_train)[:, 1]
            y_pred_bkg_val = model.predict_proba(xp_val)[ :,0 ].ravel()
            y_pred_bkg_train = model.predict_proba(xp_train)[:,0]
            fig1, ax1 = plt.subplots(1,1)
            plt.hist(y_pred_signal_val, bins=50, alpha=0.5, color='blue', label='Validation Sig')
            plt.hist(y_pred_signal_train, bins=50, alpha=0.5, color='deepskyblue', label='Training Sig')
            plt.hist(y_pred_bkg_val, bins=50, alpha=0.5, color='red', label='Validation BKG')
            plt.hist(y_pred_bkg_train, bins=50, alpha=0.5, color='firebrick', label='Training BKG')

            ax1.legend(loc="upper right")
            base_path = f"output/{name}_{year}"
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            fig1.savefig(f"output/{name}_{year}/Validation_{label}.png")
            
            y_pred = model.predict_proba(xp_val)[:, 1].ravel()
            y_pred_train = model.predict_proba(xp_train)[:, 1].ravel()
            y_eval_pred = model.predict_proba(xp_eval)[:, 1].ravel()
            print("y_pred_______________________________________________________________")
            print("y_pred_______________________________________________________________")
            print("y_pred_______________________________________________________________")
            print(y_pred)
            print("y_pred_______________________________________________________________")
            print("y_pred_______________________________________________________________")
            print("y_pred_______________________________________________________________")
            print(y_val)
            nn_fpr_xgb, nn_tpr_xgb, nn_thresholds_xgb = roc_curve(y_val.ravel(), y_pred, sample_weight=w_val)
            print(nn_fpr_xgb)
            print(nn_tpr_xgb)
            print(nn_thresholds_xgb)
            """
            for i in range(len(nn_fpr_xgb)-1):
                if(nn_fpr_xgb[i]>nn_fpr_xgb[i+1]):
                    print(i,nn_fpr_xgb[i])
                    print(i+1,nn_fpr_xgb[i+1])
                if(nn_tpr_xgb[i]>nn_tpr_xgb[i+1]):
                    print(i,nn_tpr_xgb[i])
                    print(i+1,nn_tpr_xgb[i+1])
            """
            sorted_index = np.argsort(nn_fpr_xgb)
            fpr_sorted =  np.array(nn_fpr_xgb)[sorted_index]
            tpr_sorted = np.array(nn_tpr_xgb)[sorted_index]
            #auc_xgb = auc(nn_fpr_xgb[:-2], nn_tpr_xgb[:-2])
            auc_xgb = auc(fpr_sorted, tpr_sorted)
            #auc_xgb = roc_auc_score(y_val, y_pred, sample_weight=w_val)
            print("The AUC score is:", auc_xgb)
            #plt.plot(nn_fpr_xgb, nn_tpr_xgb, marker='.', label='Neural Network (auc = %0.3f)' % auc_xgb)
            #roc_auc_gus = auc(nn_fpr_xgb,nn_tpr_xgb)
            fig, ax = plt.subplots(1,1)
            # ax.plot(nn_fpr_xgb, nn_tpr_xgb, marker='.', label='Neural Network (auc = %0.3f)' % auc_xgb)
            
            eff_bkg_train, eff_sig_train, thresholds_train = customROC_curve_AN(y_train.ravel(), y_pred_train, weight_nom_train)
            eff_bkg_val, eff_sig_val, thresholds_val = customROC_curve_AN(y_val.ravel(), y_pred, weight_nom_val)
            eff_bkg_eval, eff_sig_eval, thresholds_eval = customROC_curve_AN(y_eval.ravel(), y_eval_pred, weight_nom_eval)
            
            plt.plot(eff_sig_eval, eff_bkg_eval, label="Stage2 ROC Curve (Eval)")
            plt.plot(eff_sig_train, eff_bkg_train, label="Stage2 ROC Curve (Train)")
            plt.plot(eff_sig_val, eff_bkg_val, label="Stage2 ROC Curve (Val)")
            
            # plt.vlines(eff_sig, 0, eff_bkg, linestyle="dashed")
            plt.vlines(np.linspace(0,1,11), 0, 1, linestyle="dashed", color="grey")
            plt.hlines(np.logspace(-4,0,5), 0, 1, linestyle="dashed", color="grey")
            # plt.hlines(eff_bkg, 0, eff_sig, linestyle="dashed")
            plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.0])
            plt.xlabel('Signal eff')
            plt.ylabel('Background eff')
            plt.yscale("log")
            plt.ylim([0.0001, 1.0])
            
            plt.legend(loc="lower right")
            plt.title(f'ROC curve for ggH BDT {year}')
            fig.savefig(f"output/{name}_{year}/log_auc_{label}.png")
            plt.clf()

            # superimposed flipped log ROC start --------------------------------------------------------------------------
            plt.plot(1-eff_sig_eval, 1-eff_bkg_eval, label="Stage2 ROC Curve (Eval)")
            plt.plot(1-eff_sig_train, 1-eff_bkg_train, label="Stage2 ROC Curve (Train)")
            plt.plot(1-eff_sig_val, 1-eff_bkg_val, label="Stage2 ROC Curve (Val)")
            
            # plt.vlines(eff_sig, 0, eff_bkg, linestyle="dashed")
            plt.vlines(np.linspace(0,1,11), 0, 1, linestyle="dashed", color="grey")
            plt.hlines(np.logspace(-4,0,5), 0, 1, linestyle="dashed", color="grey")
            # plt.hlines(eff_bkg, 0, eff_sig, linestyle="dashed")
            plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.0])
            plt.xlabel('1 - Signal eff')
            plt.ylabel('1- Background eff')
            plt.yscale("log")
            plt.ylim([0.0001, 1.0])
            
            plt.legend(loc="lower right")
            plt.title(f'ROC curve for ggH BDT {year}')
            fig.savefig(f"output/{name}_{year}/logFlip_auc_{label}.png")
            fig.savefig(f"output/{name}_{year}/logFlip_auc_{label}.pdf")
            plt.clf()
            # superimposed flipped log ROC end --------------------------------------------------------------------------

            

            # results = model.evals_result()
            # print(results.keys())
            # plt.plot(results['validation_0']['logloss'], label='train')
            # plt.plot(results['validation_1']['logloss'], label='test')
            # # show the legend
            # plt.legend()
            # plt.savefig(f"output/{name}_{year}/Loss_{label}.png")

            # feature_important = model.get_booster().get_score(importance_type='gain')
            # keys = list(feature_important.keys())
            # values = list(feature_important.values())

            # data = pd.DataFrame(data=values, index=training_features, columns=["score"]).sort_values(by = "score", ascending=True)
            # data.nlargest(50, columns="score").plot(kind='barh', figsize = (20,10))
            # plt.savefig(f"test.png")
            labels = [feat.replace("_nominal","") for feat in training_features]
            model.get_booster().feature_names = labels 
            
            feature_important = model.get_booster().get_score(importance_type='weight')
            keys = list(feature_important.keys())
            values = list(feature_important.values())
            score_name = "Weight Score"
            data = pd.DataFrame(data=values, index=keys, columns=[score_name]).sort_values(by = score_name, ascending=True)
            data.nlargest(50, columns=score_name).plot(kind='barh', figsize = (20,10))
            data.plot(kind='barh', figsize = (20,10))
            plt.savefig(f"output/{name}_{year}/BDT_FeatureImportance_{label}_byWeight.png")
    
            feature_important = model.get_booster().get_score(importance_type='gain')
            keys = list(feature_important.keys())
            values = list(feature_important.values())
            score_name = "Gain Score"
            data = pd.DataFrame(data=values, index=keys, columns=[score_name]).sort_values(by = score_name, ascending=True)
            data.nlargest(50, columns=score_name).plot(kind='barh', figsize = (20,10))
            data.plot(kind='barh', figsize = (20,10))
            plt.savefig(f"output/{name}_{year}/BDT_FeatureImportance_{label}_byGain.png")



            
            #save('x_val_{label}.npy', x_val[training_features])
            #save('y_val_{label}.npy', y_val)
            #save('weight_val_{label}.npy', df_val['training_wgt'].values)
            output_path = args["output_path"]
            #util.save(history.history, f"output/trained_models/history_{label}_bdt.coffea") 
            base_path = f"{output_path}/{name}_{year}"
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            model_fname = (f"{base_path}/{name}_{label}.pkl")
            pickle.dump(model, open(model_fname, "wb"))
            print ("wrote model to",model_fname)
            

def evaluation(df, args):
    if df.shape[0]==0: return df
    if args['dnn']:
        import keras.backend as K
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        config = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            allow_soft_placement=True,
            device_count = {'CPU': 1})
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        sess = tf.compat.v1.Session(config=config)
    
        if args['do_massscan']:
            mass_shift = args['mass']-125.0
        add_year = args['evaluate_allyears_dnn']
        #df = prepare_features(df, args, add_year)
        df['dnn_score'] = 0
        with sess:
            nfolds = 4
            for i in range(nfolds):
                if args['evaluate_allyears_dnn']:
                    label = f"allyears_{args['label']}_{i}"
                else:
                    label = f"{args['year']}_{args['label']}{i}"

                train_folds = [(i+f)%nfolds for f in [0,1]]
                val_folds = [(i+f)%nfolds for f in [2]]
                eval_folds = [(i+f)%nfolds for f in [3]]
                
                eval_filter = df.event.mod(nfolds).isin(eval_folds)
                
                scalers_path = f'output/trained_models/scalers_{label}.npy'
                scalers = np.load(scalers_path)
                model_path = f'output/trained_models/test_{label}.h5'
                dnn_model = load_model(model_path)
                df_i = df[eval_filter]
                #if args['r']!='h-peak':
                #    df_i['dimuon_mass'] = 125.
                #if args['do_massscan']:
                #    df_i['dimuon_mass'] = df_i['dimuon_mass']+mass_shift

                df_i = (df_i[training_features]-scalers[0])/scalers[1]
                prediction = np.array(dnn_model.predict(df_i)).ravel()
                df.loc[eval_filter,'dnn_score'] = np.arctanh((prediction))
    if args['bdt']:
        import xgboost as xgb
        import pickle
        if args['do_massscan']:
            mass_shift = args['mass']-125.0
        add_year = args['evaluate_allyears_dnn']
        #df = prepare_features(df, args, add_year)
        df['bdt_score'] = 0
        nfolds = 4
        for i in range(nfolds):    
            if args['evaluate_allyears_dnn']:
                label = f"allyears_{args['label']}_{i}"
            else:
                label = f"{args['year']}_{args['label']}{i}"
            eval_label = f"{args['year']}_{args['label']}0"
            train_folds = [(i+f)%nfolds for f in [0,1]]
            val_folds = [(i+f)%nfolds for f in [2]]
            eval_folds = [(i+f)%nfolds for f in [3]]
            
            eval_filter = df.event.mod(nfolds).isin(eval_folds)
            output_path = args["output_path"]
            scalers_path = f"{output_path}/{name}_{year}/scalers_{name}_{eval_label}.npy"
            #scalers_path = f'output/trained_models_nest10000/scalers_{label}.npy'
            scalers = np.load(scalers_path)
            model_path = f"{output_path}/{name}_{year}/{name}_{label}.pkl"
            #model_path = f'output/trained_models_nest10000/BDT_model_earlystop50_{label}.pkl'
            bdt_model = pickle.load(open(model_path, "rb"))
            print(bdt_model.classes_)
            df_i = df[eval_filter]
            #if args['r']!='h-peak':
            #    df_i['dimuon_mass'] = 125.
            #if args['do_massscan']:
            #    df_i['dimuon_mass'] = df_i['dimuon_mass']+mass_shift
            #print("Scaler 0 ",scalers[0])
            #print("Scaler 1 ",scalers[1])
            #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",df_i.head())
            df_i = (df_i[training_features]-scalers[0])/scalers[1]
            #print("************************************************************",df_i.head())
            prediction_sig = np.array(bdt_model.predict_proba(df_i.values)[:, 1]).ravel()
            prediction_bkg = np.array(bdt_model.predict_proba(df_i.values)[:, 0]).ravel()
            fig1, ax1 = plt.subplots(1,1)
            plt.hist(prediction_sig, bins=50, alpha=0.5, color='blue', label='Validation Sig')
            
            plt.hist(prediction_bkg, bins=50, alpha=0.5, color='red', label='Validation BKG')


            ax1.legend(loc="upper right")
            base_path = f"output/{name}_{year}"
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            fig1.savefig(f"output/{name}_{year}/Validation_{label}.png")
            
            

            # df.loc[eval_filter,'bdt_score'] = prediction
    return df
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="Year to process (2016preVFP, 2016postVFP, 2017 or 2018)",
    )
    parser.add_argument(
    "-n",
    "--name",
    dest="name",
    default="test",
    action="store",
    help="Name of classifier",
    )
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default="/depot/cms/users/yun79/results/stage1/DNN_test2/",
    action="store",
    help="Year to process (2016preVFP, 2016postVFP, 2017 or 2018)",
    )
    
    sysargs = parser.parse_args()
    year = sysargs.year
    name = sysargs.name
    args = {
        "dnn": False,
        "bdt": True,
        "year": year,
        "name": name,
        "do_massscan": False,
        "evaluate_allyears_dnn": False,
        "output_path": "/depot/cms/users/yun79/hmm/trained_MVAs",
        "label": ""
    }
    start_time = time.time()
    client =  Client(n_workers=31,  threads_per_worker=1, processes=True, memory_limit='4 GiB') 
    load_path = f"{sysargs.load_path}/{year}/" # copperheadV1
    print(f"load_path: {load_path}")
    sample_l = training_samples["background"] + training_samples["signal"]

    

    fields2load = [
        "dimuon_mass",
        "jj_mass_nominal",
        "jj_dEta_nominal",
        "jet1_pt_nominal",
        "nBtagLoose_nominal",
        "nBtagMedium_nominal",
        "mmj1_dEta_nominal",
        "mmj2_dEta_nominal",
        "mmj1_dPhi_nominal",
        "mmj2_dPhi_nominal",
        "wgt_nominal",
        "dimuon_ebe_mass_res",
        "event",
        "mu1_pt",
        "mu2_pt",
        "region",
    ]
    fields2load = list(set(fields2load + training_features)) # remove redundancies


    df_l = []
    print(f"sample_l: {sample_l}")
    print(f"training_features: {training_features}")
    for sample in sample_l:
        zip_sample = dak.from_parquet(load_path+f"/{sample}/*.parquet")
        computed_zip = ak.zip({
            field : zip_sample[field] for field in fields2load
        }).compute()
        # computed_dict = {field: (zip_sample[field]) for field in fields2load}
        computed_dict = {}
        for field in computed_zip.fields:
            if "dPhi" in field:
                computed_dict[field] = (ak.fill_none(computed_zip[field], value=-1.0))
            elif "region" in field:
                computed_dict[field] = list(computed_zip[field])
            else:
                computed_dict[field] = (ak.fill_none(computed_zip[field], value=0.0))
            # print(f"computed_dict[{field}] : {computed_dict[field]}")
        df_sample = pd.DataFrame(computed_dict) # direct ak.to_dataframe doesn't work for some reason
        df_sample["dataset"] = sample # add back the dataset column
        df_l.append(df_sample)
        # print(f"{sample} df: {df_sample}")
    df = pd.concat(df_l)
    # overwrite mu over mass variables, they're all inf or zeros for some reason
    
    df["mu1_pt_over_mass"] = df["mu1_pt"] / df["dimuon_mass"]
    df["mu2_pt_over_mass"] = df["mu2_pt"] / df["dimuon_mass"]
    # print(df['dataset'].unique())
    mu1_pt_over_mass = df["mu1_pt_over_mass"]
    mu2_pt_over_mass = df["mu2_pt_over_mass"]
    print(f"mu1_pt_over_mass: {mu1_pt_over_mass}")
    print(f"mu2_pt_over_mass: {mu2_pt_over_mass}")
    
    #df_signal = pd.read_pickle("/depot/cms/hmm/purohita/coffea/my_training_signal_sample.pickle")
    #df_background = pd.read_pickle("/depot/cms/hmm/purohita/coffea/my_training_sample.pickle")
    #df = pd.concat([df_signal, df_background])
    
    #df = pd.read_pickle("/depot/cms/hmm/coffea/training_dataset.pickle")
    # df = pd.read_pickle(f"/depot/cms/hmm/vscheure/training_dataset_ggHnew_{year}.pickle")
    #df = pd.read_pickle("/depot/cms/hmm/purohita/coffea/training_dataset100FromEach_may14.pickle")
    #df = pd.read_pickle("/depot/cms/hmm/purohita/coffea/my_training_signal_sample.pickle")
    #df = pd.read_pickle("/depot/cms/hmm/purohita/coffea/my_training_sample.pickle")
    # print(df)
    
    #df = df[:100000]
    #df = df.sample(frac = 0.1)
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df = df.dropna()
    #df = df.fillna(0)
    #print(np.isinf(df).any())
    """
    for col in training_features:
        print(col)
        print(np.isinf(df[col]).values.sum())
        print(df[col].isnull().sum())
    """
    #split_into_channels(df, "nominal")
    #df["channel"] = df["channel_nominal"]
   
    vbf_filter = (
        (df[f"nBtagLoose_nominal"] < 2) &
        (df[f"nBtagMedium_nominal"] < 1) &
        (df[f"jj_mass_nominal"] > 400) &
        (df[f"jj_dEta_nominal"] > 2.5) &
        (df[f"jet1_pt_nominal"] > 35)
    )
    df = df[vbf_filter==False]

    """
    ggh_filter = (((df['njets_nominal'] == 0) | 
                  ((df['njets_nominal'] == 1) & ((df['jj_mass_nominal'] <= 400) | (df['jj_dEta_nominal'] <= 2.5) | (df['jet1_pt_nominal']< 35))) | 
                  ((df['njets_nominal'] >= 2) & ((df['jj_mass_nominal'] <= 400) | (df['jj_dEta_nominal'] <= 2.5) | (df['jet1_pt_nominal']< 35)))
               ) & ((df['dimuon_mass'] > 115) & (df['dimuon_mass'] < 135)))
    """
    # print(df['njets_nominal'])
    df['njets_nominal']= df['njets_nominal'].fillna(0)
    # print(df['njets_nominal'])
    ggh_filter = (  (df[f"nBtagLoose_nominal"] < 2) &
                    (df[f"nBtagMedium_nominal"] < 1) &
                    #(df['njets_nominal'] > 0) & #temp for testing
                 ((df['dimuon_mass'] > 115) & (df['dimuon_mass'] < 135)))
    df = df[ggh_filter]
    #print(df.shape)
    df['wgt_nominal'] = np.abs(df['wgt_nominal'])
    print("aftergghfiletr")

    df = prepare_dataset(df[df['region']=='h-peak'], training_samples)
    #print(df["class"])
    #print([ x for x in df.columns if "nominal" in x])
    # print(df[df['dataset']=="dy_M-100To200"])
    # print(df[df['dataset']=="ggh_powheg"])
    """
    for col in training_features:
        print(col)
        print(np.isinf(df[col]).values.sum())
        print(df[col].isnull().sum())
    """
    classifier_train(df, args)
    evaluation(df, args)
    #df.to_pickle('/depot/cms/hmm/purohita/coffea/eval_dataset.pickle')
    #print(df)




"""
test_bdt = XGBClassifier(
    max_depth=7,  # for 2018
    # max_depth=6,previous value
    n_estimators=100,
    # n_estimators=100,
    # objective='multi:softmax',
    objective="binary:logistic",
    num_class=1,
    # learning_rate=0.001,#for 2018
    # learning_rate=0.0034,#previous value
    # reg_alpha=0.680159426755822,
    # colsample_bytree=0.47892268305051233,
    min_child_weight=20,
    # subsample=0.5606,
    # reg_lambda=16.6,
    # gamma=24.505,
    # n_jobs=5,
    tree_method="hist",
)
"""
