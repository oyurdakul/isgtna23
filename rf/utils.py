from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler


import pandas as pd
import numpy as np
import math
from datetime import date
import datetime
import csv
import time
import os
import json
import shutil, errno

# Plots
# ==============================================================================
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
# Turn interactive plotting off
plt.ioff()



def prepare_forecast_data(ts_data,target_data,n_scenarios,load_scaling = 1,n_oos=1, start_date_train = False, start_date_oos = False):
    date_range = ts_data.index.date

    ## dataset size by number of training days
    n_train_days = n_scenarios
    if start_date_train:
        start_in = np.where(date_range == date.fromisoformat(start_date_train)) 
        start_i = start_in[0][0]
    else:
        start_i = np.where(date_range == date.fromisoformat('2018-05-15'))[0][0]
    start_train = str(date_range[start_i]) + ' 23:55:00'
    end_train = str(date_range[start_i + n_train_days]) + ' 23:55:00'

    if start_date_oos:
        start_oos = np.where(date_range == date.fromisoformat(start_date_oos)) 
        start_o = start_oos[0][0]
    else:
        start_o = np.where(date_range == date.fromisoformat('2019-07-15'))[0][0]
    start_test = str(date_range[start_o]) + ' 23:55:00'
    end_test = str(date_range[start_o + n_oos]) + ' 23:55:00'


    target_data = target_data * load_scaling

    X_train = ts_data.loc[start_train:end_train, :].values
    y_train = target_data.loc[start_train:end_train, :].values
    X_test  = ts_data.loc[start_test:end_test, :].values
    y_test  = target_data.loc[start_test:end_test, :].values


    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaler = X_scaler.fit(X_train)
    y_scaler = y_scaler.fit(y_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)


    if len(X_train) != len(y_train):
        raise ValueError('Different feature and label sizes in the train set')
        
    if len(X_test) != len(y_test):
        raise ValueError('Different feature and label sizes in the test set')
    
    return X_train, X_test,y_train, y_test, X_scaler, y_scaler

# retrieve the number of training samples that end up in each leaf of an estimator
def get_n_samples_in_leaf_of_estimator(num_trees,train_leaves):
    n_samples_in_leaf_of_estimator = []
    for t in range(num_trees):
        unique, counts = np.unique(train_leaves[:,t], return_counts=True)
        d = dict(zip(unique, counts))
        n_samples_in_leaf_of_estimator.append(d)
    return n_samples_in_leaf_of_estimator

# retrieve the weights of a new arriving observation
def get_model_weights(model,num_trees,n_samples_in_leaf_of_estimator,x_in,train_leaves):
    weights = dict()
    tree_weights = np.zeros(train_leaves.shape[0])
    leaf_id_pred = model.apply(x_in).flatten()
    for i,leaf_id_x_i in enumerate(train_leaves):
        for t in range(num_trees):
            if  leaf_id_x_i[t] == leaf_id_pred[t]:
                tree_weights = 1/n_samples_in_leaf_of_estimator[t][leaf_id_pred[t]]    
                tree_weights /= num_trees   
                if i in weights.keys():
                    w = weights[i] + tree_weights
                else:
                    w = tree_weights
                weights[i] = w
    return weights

# get the number of estimators in the ensemble
def get_number_of_trees(model):

    if isinstance(model,(RandomForestRegressor)):
        num_trees = model.n_estimators
    return num_trees

# check if weights add up to 1
def check_sum(weights):
    if np.round(sum(weights.values()),2) != 1:
        raise ValueError('weights do not sum to 1')

# fill the weights of scenarios with a probability of 0
def fill_weights(weights,X_train):
    for i,x in enumerate(X_train):
        if i not in weights.keys():
            weights[i] = 0
    return weights  

# if necessary multiply by the learning rate used for training the model
def multiply_by_lr(weights,hp):
    if hp['learning_rate'] is math.nan:
        return weights
    else:
        weights.update({k: v * hp['learning_rate'] for k, v in weights.items()})
        return weights

# if the first estimator of the ensemble only contains a single leaf, add 1/D to the weights
def account_for_single_leaf(weights,D):
    weights.update({k: v + (1/D) for k, v in weights.items()})
    return weights

# manipulate the weights dependend on the traning set size
def manipulate_weights(weights, Xi):
    weights.update({k:v**(Xi) for k,v in weights.items()})
    return weights

def normalize_weights(weights):
    print(sum(weights.values()))
    weights.update({k:v/sum(weights.values()) for k,v in weights.items()}) 
    for k in weights.keys():
        if weights[k]<10**-6:
            weights[k]=5*10**-3
    weights.update({k:v/sum(weights.values()) for k,v in weights.items()}) 
    return weights


def weights_processing(model,weights,X_train,hp,D):
    weights = fill_weights(weights,X_train)
    weights = multiply_by_lr(weights,hp)
 
    xi = hp['Xi']
    weights = normalize_weights(weights)
    weights = manipulate_weights(weights,xi)
    weights = normalize_weights(weights)
    check_sum(weights)
    return weights


def get_model_policy(model,X_train,X_test,hp):
    D = X_train.shape[0]
    num_trees = get_number_of_trees(model)
    train_leaves = model.apply(X_train)
    n_samples_in_leaf_of_estimator = get_n_samples_in_leaf_of_estimator(num_trees,train_leaves)
    weights = get_model_weights(model,num_trees,n_samples_in_leaf_of_estimator,X_test,train_leaves)
    weights = weights_processing(model,weights,X_train,hp,D)
    return weights

def train_random_forest_model_with_hp(hp,X_train,y_train):
    n_estimators = hp['n_estimators']
    max_depth = hp['max_depth']
    min_samples_split = hp['min_samples_split']
    max_features = hp['max_features']
    rf = RandomForestRegressor(n_estimators=n_estimators,
                                max_depth=max_depth,
                                # min_samples_split = min_samples_split,
                                max_features = max_features)
    rf.fit(X_train,y_train)
    return rf


def train_model_with_hp(model,hyperparameters,X_train,y_train):

    if model == 'rf':
        model = train_random_forest_model_with_hp(hyperparameters,X_train,y_train)

    # elif model == 'gbm':
    #     model = train_gradient_boost_model_with_hp(hyperparameters,X_train,y_train)

 
    return model

def train_model_with_hp_pp(model,hyperparameters,X_train,y_train):

    if model == 'rf':
        model = train_random_forest_model_with_hp(hyperparameters,X_train,y_train)


    return model

def make_prediction(model, X_test, y_test,return_prediction = False):
    y_hat = model.predict(X_test)
    print(model.__repr__)
    mse = mean_squared_error(y_hat.flatten(),y_test.flatten())
    r2 = model.score(X_test,y_test)
    mape = mean_absolute_percentage_error(y_hat.flatten(),y_test.flatten())
    print('MSE on test set: ', mse)
    print('MAPE on test set: ', mape)
    print('R2 score on test set: ', r2)
    if return_prediction:
        return y_hat, mse, mape,r2

def write_oos_loads_to_json(json_data, y_test,n,filename,y_scaler):
    buses = json_data['Buses'].keys()
    n_buses = len(json_data['Buses'].keys())
    load = y_scaler.inverse_transform(y_test[n-1].reshape(1,-1))
    load = np.round(load,2).reshape(-1)
    b_i = 0
    for b in buses:
        if json_data['Buses'][b]['s1']['Load (MW)'] != [0]*24: 
            bus_load = load[b_i * 24 : (b_i + 1) * 24]
            json_data['Buses'][b]['s1']['Load (MW)'] = bus_load.tolist()
            b_i += 1
            
    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=3)

def write_5min_oos_loads_to_json(json_data, y_test,n,filename,y_scaler):
    json_data["Parameters"]["Time step (min)"] = 5
    buses = json_data['Buses'].keys()
    n_buses = len(json_data['Buses'].keys())
    load = y_scaler.inverse_transform(y_test[n-1].reshape(1,-1))
    load = np.round(load,2).reshape(-1)
    b_i = 0
    for b in buses:
        if json_data['Buses'][b]['s1']['Load (MW)'] != [0]*288: 
            bus_load = load[b_i * 288: (b_i + 1) * 288]
            json_data['Buses'][b]['s1']['Load (MW)'] = bus_load.tolist()
            b_i += 1
            
    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=3)

def write_scenario_loads_to_json(json_data, y_train, pol, filename,y_scaler):
    buses = json_data['Buses'].keys()
    n_buses = len(json_data['Buses'].keys())
    scenarios = json_data['Buses']['b1'].keys()
    
    for s_i, s in enumerate(scenarios):
        # write scenario loads
        load = y_scaler.inverse_transform(y_train[s_i].reshape(1,-1))
        load = np.round(load,2).reshape(-1)
        b_i = 0
        for b in buses:
            if json_data['Buses'][b][s]['Load (MW)'] != [0]*24: 
                bus_load = load[b_i * 24 : (b_i + 1) * 24]
                json_data['Buses'][b][s]['Load (MW)'] = bus_load.tolist()
                b_i += 1
            json_data['Buses'][b][s]['Probability'] = float(np.round(pol[s_i],10))

    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=3)

def write_naive_scenario_loads_to_json(json_data, y_train, filename,y_scaler,n_sns):
    buses = json_data['Buses'].keys()
    n_buses = len(json_data['Buses'].keys())
    scenarios = json_data['Buses']['b1'].keys()
    
    for s_i, s in enumerate(scenarios):
        # write scenario loads
        load = y_scaler.inverse_transform(y_train[s_i].reshape(1,-1))
        load = np.round(load,2).reshape(-1)
        b_i = 0
        for b in buses:
            if json_data['Buses'][b][s]['Load (MW)'] != [0]*24: 
                bus_load = load[b_i * 24 : (b_i + 1) * 24]
                json_data['Buses'][b][s]['Load (MW)'] = bus_load.tolist()
                b_i += 1
            json_data['Buses'][b][s]['Probability'] = float(np.round(1/n_sns,10))

    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=3)

def write_scenario_loads_to_json_for_point_prediction(json_data,y_hat,filename,y_scaler):
    buses = json_data['Buses'].keys()
    n_buses = len(json_data['Buses'].keys())
    scenarios = json_data['Buses']['b1'].keys()
    
    for s_i, s in enumerate(scenarios):
        # write scenario loads
        load = y_scaler.inverse_transform(y_hat[s_i].reshape(1,-1))
        load = np.round(load,2).reshape(-1)
        b_i = 0
        for b in buses:
            if json_data['Buses'][b][s]['Load (MW)'] != [0]*24: 
                bus_load = load[b_i * 24 : (b_i + 1) * 24]
                json_data['Buses'][b][s]['Load (MW)'] = bus_load.tolist()
                b_i += 1
            
            json_data['Buses'][b][s]['Probability'] = 1.0

    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=3)


def read_costs_from_json(model,n_scenarios,n_oos_samples,OUTPUT_FILE_DIR):
    filename = OUTPUT_FILE_DIR + f'{model}/{str(n_scenarios)}/oos_outputs/{n_oos_samples}/oos_costs.txt'
    with open(filename) as f:
        lines = f.readlines()
    costs = [float(line.strip()) for line in lines]
    return costs

def append_results_to_csv(result,DIR,n_oos):
    DST_DIR = 'experiments/'
    with open(DST_DIR + f'results_n_oos_{n_oos}.csv','a') as f:
        wr = csv.writer(f)
        wr.writerow(result)


def plot_commitment_status(model,n_scenarios,n_oos_samples,hp,OUTPUT_FILE_DIR,use_weights):
    plt.figure(figsize=(10,5))
    filename = OUTPUT_FILE_DIR + f'{model}/{str(n_scenarios)}/oos_outputs/{n_oos_samples}/{n_oos_samples}.json'
    with open(filename) as json_file:
        json_data = json.load(json_file)
    g_comm = json_data['Is on']
    l_com = []
    for k in g_comm.keys():
        g = g_comm[k]
        l_com.append(g)
    com_df = pd.DataFrame(l_com)
    com_df.index = g_comm.keys()

    # Define colors
    colors = ["gray", "green"] 
    cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))
    # Plot the data
    vmin, vmax = 0,1
    ax = sns.heatmap(com_df,cmap=cmap, linewidths=1, linecolor='white',vmin=vmin,vmax=vmax,center= (vmin+vmax)/2)
    # Set the colorbar labels
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.25,0.75])
    colorbar.set_ticklabels(['0', '1'])
    plt.savefig(OUTPUT_FILE_DIR + f"{model}/{str(n_scenarios)}/oos_outputs/{n_oos_samples}/{model}-{n_scenarios}-{use_weights}-{hp['n_estimators']}-{hp['max_depth']}-{hp['learning_rate']}-{n_oos_samples}-commitment_status.png")

def run_hp_experiment(m, hp, X_train, X_test, y_train,n_oos,n_scenarios,y_scaler):
    print(f"\n=== starting experiment with {m} model ===")
    print(f"--- number of scenarios {n_scenarios} ---")
    print(f"--- n_estimators {hp['n_estimators']} ---")
    print(f"--- max_depth {hp['max_depth']} ---")
    print(f"--- learning_rate {hp['learning_rate']} ---")
    INPUT_FILE_DIR = 'input_files/'
    start = time.time()
    model = train_model_with_hp(m,hp, X_train, y_train)

    for o in range(1,n_oos+1):
        n_es = str(hp['n_estimators'])
        xi = str(hp['Xi'])
        m_d = str(hp['max_depth'])
        l_r = str(hp['learning_rate'])
        m_s_l = str(hp['min_split_loss'])
        m_s_s = str(hp['min_samples_split'])
        m_f = str(hp['max_features'])
        filename = INPUT_FILE_DIR + f'snum_{str(n_scenarios)}/oos_{str(o)}/weighted/{str(m)}/n_est_{n_es}_Xi_{xi}_max_depth_{m_d}_lear_rate_{l_r}_min_sp_l_{m_s_l}_min_samples_split_{m_s_s}_max_features_{m_f}.json'
        with open(filename) as json_file:
            json_data = json.load(json_file)
        print('--- oos_scenario: ', o, ' ---')
        x_in = X_test[o-1].reshape(1, -1)
        pol = get_model_policy(model,X_train,x_in,hp)
        write_scenario_loads_to_json(json_data,y_train,pol,filename,y_scaler)
    end = time.time()

    print(f"=== experiment finished in {np.round(end-start,2)} s===")

def run_hp_experiment_for_point_prediction(m, hp, X_train, X_test, y_train,n_oos,n_scenarios,y_scaler):
    INPUT_FILE_DIR = 'input_files/'
    print(f"\n=== starting experiment with {m} model ===")
    print(f"--- n_estimators {hp['n_estimators']} ---")
    print(f"--- max_depth {hp['max_depth']} ---")
    print(f"--- learning_rate {hp['learning_rate']} ---")
    start = time.time()

    model = train_model_with_hp_pp(m,hp, X_train, y_train)
    
    for o in range(1,n_oos+1):
        n_es = str(hp['n_estimators'])
        m_d = str(hp['max_depth'])
        l_r = str(hp['learning_rate'])
        m_s_l = str(hp['min_split_loss'])
        m_s_s = str(hp['min_samples_split'])
        m_f = str(hp['max_features'])
        filename = INPUT_FILE_DIR + f'snum_{str(n_scenarios)}/oos_{str(o)}/point/{str(m)}/n_est_{n_es}_Xi_xi_max_depth_{m_d}_lear_rate_{l_r}_min_sp_l_{m_s_l}_min_samples_split_{m_s_s}_max_features_{m_f}.json'
        with open(filename) as json_file:
            json_data = json.load(json_file)
        print('--- oos_scenario: ', o, ' ---')
        x_in = X_test[o-1].reshape(1, -1)
        y_hat = model.predict(x_in)
        write_scenario_loads_to_json_for_point_prediction(json_data,y_hat,filename,y_scaler)
       
    end = time.time()

    print(f"=== experiment finished in {np.round(end-start,2)} s===")

def run_naive_benchmark(m, y_train,n_oos,n_scenarios,DIR,OUTPUT_FILE_DIR,Main,y_scaler):
    INPUT_FILE_DIR = 'input_files/'
    print(f"\n=== starting experiment with {m} model ===")
    start = time.time()

    for o in range(1,n_oos+1):
        filename = INPUT_FILE_DIR + f'/weights/{m}/{str(o)}_{str(n_scenarios)}.json'
        with open(filename) as json_file:
            json_data = json.load(json_file)
        write_naive_scenario_loads_to_json(json_data, y_train, filename,y_scaler,n_scenarios)
        print('--- oos_scenario: ', o, ' ---')
        # run julia script
        print('--- run julia script ---')
        Main.scen_num = n_scenarios
        Main.model_type = m 
        Main.n_oos = n_oos
        Main.use_weights = True
        Main.n_oos_i = o
        Main.eval("mainfunc(model_type,scen_num,n_oos,n_oos_i,use_weights)")
        # get cost from outputfile
        exp_costs = read_costs_from_json(m,n_scenarios,n_oos,OUTPUT_FILE_DIR)
        print(f"--- exp_costs {exp_costs[o-1]} ---")
        hp  = {}
        hp['n_estimators'] = math.nan
        hp['max_depth'] = math.nan
        hp['learning_rate'] = math.nan
        # plot_commitment_status(m,n_scenarios,o,hp,OUTPUT_FILE_DIR,use_weights=False)
        # append costs to result file
        result = [datetime.datetime.now(),m, n_scenarios,math.nan,math.nan,math.nan,False] + [exp_costs[o-1]]

        append_results_to_csv(result,DIR,o)
        save_results(OUTPUT_FILE_DIR,True,m,n_scenarios,n_oos)
    end = time.time()

    print(f"=== experiment finished in {np.round(end-start,2)} s===")

def test_models(m,hp, X_train, y_train, X_test, y_test, n_oos,y_scaler,n_sns):
    model = train_model_with_hp_pp(m,hp, X_train, y_train)
    res_f = 'model_tests.csv'
    for o in range(1,n_oos+1):
        print('--- oos_scenario: ', o, ' ---')
        x_in = X_test[o-1].reshape(1, -1)
        y_hat = model.predict(x_in)
        y_hat = y_hat[0]
        mse = mean_squared_error(y_test[o-1], y_hat)
        print(f"--- mse {mse} ---")
        mae = mean_absolute_error(y_test[o-1], y_hat)
        print(f"--- mae {mae} ---")
        res = [m,n_sns,o,mae,mse]
        with open(res_f,'a') as f:
            wr = csv.writer(f)
            wr.writerow(res)

def test_naive(X_train, y_train, X_test, y_test, n_oos,n_sns):
    res_f = 'model_tests.csv'
    y_hat = np.mean(y_train,axis=0)
    for o in range(1,n_oos+1):
        print('--- oos_scenario: ', o, ' ---')
        mse = mean_squared_error(y_test[o-1], y_hat)
        print(f"--- mse {mse} ---")
        mae = mean_absolute_error(y_test[o-1], y_hat)
        print(f"--- mae {mae} ---")
        res = ['naive',n_sns,o,mae,mse]
        with open(res_f,'a') as f:
            wr = csv.writer(f)
            wr.writerow(res)


def save_results(OUTPUT_FILE_DIR,use_weights,model,n_scenarios,n_oos):
    DST_DIR = 'experiments/'
    src = OUTPUT_FILE_DIR + f'{model}/{n_scenarios}/oos_outputs/{n_oos}/'
    if use_weights:
        dst = DST_DIR + '/output_files/weights/'
    else:
        dst = DST_DIR + '/output_files/point_prediction/' 
    dst = dst + f'{model}/{n_scenarios}/oos_outputs/{n_oos}/'
    copy_results(src,dst)

def copy_results(src, dst):
    try:
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    except OSError as exc: 
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise
 
# create empty results files
def create_results_files(OUTPUT_DIR):
    exp = str(datetime.datetime.now()).replace(' ','_')
    exp = exp.replace(':','-')

    DIR = OUTPUT_DIR +f'{exp}'
    os.mkdir(DIR)
    for n_oos in range(1,21):
        df = pd.DataFrame(columns=['model','n_scenarios','n_estimators','max_depth','learning_rate','use_weights',f'cost_oos'])
        df.to_csv(DIR + f'results_n_oos_{n_oos}.csv',sep=',')
    return DIR

def mk_exp_dir(OUTPUT_FILE_DIR, MODEL, n_scenarios, n_oos):
    exp_dir = OUTPUT_FILE_DIR + '/' + MODEL + '/' + str(n_scenarios) + '/oos_outputs' + '/' + str(n_oos) + '/'
    os.makedirs(exp_dir,exist_ok=True)

def assign_params(MODEL, params):
    n_oos = params['n_oos']
    hp_search_space = params[MODEL]
    return hp_search_space, n_oos

def create_new_scenario_files_point(REF_FILE, NEW_FILE_PATH, MODEL, n_sns, hp):
    filename = REF_FILE
    with open(filename) as json_file:
        json_data = json.load(json_file)
    json_data['Parameters']['Scenario number'] = n_sns
    buses = list(json_data['Buses'].keys())
    
    for b in buses:
        copy_data = json_data['Buses'][b]['s1']
        for s in range(2, n_sns + 1):
            json_data['Buses'][b]['s{}'.format(s)] = copy_data
    for n_estimators in hp['n_estimators']:
        for max_depth in hp['max_depth']:
            for learning_rate in hp['learning_rate']:
                for min_split_loss in hp['min_split_loss']:
                    for min_samples_split in hp['min_samples_split']:
                        for max_features in hp['max_features']:
                            n_es = str(n_estimators)
                            m_d = str(max_depth)
                            l_r = str(learning_rate)
                            m_s_l = str(min_split_loss)
                            m_s_s = str(min_samples_split)
                            m_f = str(max_features)
                            filename = NEW_FILE_PATH + f'n_est_{n_es}_Xi_xi_max_depth_{m_d}_lear_rate_{l_r}_min_sp_l_{m_s_l}_min_samples_split_{m_s_s}_max_features_{m_f}.json'
                            with open(filename, 'w', encoding='utf-8') as f:
                                    json.dump(json_data, f, ensure_ascii=False, indent=3)

def create_new_scenario_files(REF_FILE, NEW_FILE_PATH, MODEL, n_sns, hp):
    filename = REF_FILE
    with open(filename) as json_file:
        json_data = json.load(json_file)
    json_data['Parameters']['Scenario number'] = n_sns
    buses = list(json_data['Buses'].keys())
    
    for b in buses:
        copy_data = json_data['Buses'][b]['s1']
        for s in range(2, n_sns + 1):
            json_data['Buses'][b]['s{}'.format(s)] = copy_data
    for n_estimators in hp['n_estimators']:
        for Xi in hp['Xi']:
            for max_depth in hp['max_depth']:
                for learning_rate in hp['learning_rate']:
                    for min_split_loss in hp['min_split_loss']:
                        for min_samples_split in hp['min_samples_split']:
                            for max_features in hp['max_features']:
                                n_es = str(n_estimators)
                                xi = str(Xi)
                                m_d = str(max_depth)
                                l_r = str(learning_rate)
                                m_s_l = str(min_split_loss)
                                m_s_s = str(min_samples_split)
                                m_f = str(max_features)
    

                                filename = NEW_FILE_PATH + f'n_est_{n_es}_Xi_{xi}_max_depth_{m_d}_lear_rate_{l_r}_min_sp_l_{m_s_l}_min_samples_split_{m_s_s}_max_features_{m_f}.json'

                                with open(filename, 'w', encoding='utf-8') as f:
                                        json.dump(json_data, f, ensure_ascii=False, indent=3)

def create_naive_files(REF_FILE, NEW_FILE_PATH, n_sns):
    filename = REF_FILE
    with open(filename) as json_file:
        json_data = json.load(json_file)
    json_data['Parameters']['Scenario number'] = n_sns
    buses = list(json_data['Buses'].keys())
    for b in buses:
        copy_data = json_data['Buses'][b]['s1']
        for s in range(2, n_sns + 1):
            json_data['Buses'][b]['s{}'.format(s)] = copy_data

    filename = NEW_FILE_PATH + f'naive.json'

    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=3)

def create_oos_files(REF_FILE, NEW_FILE_PATH, oos_n):
    filename = REF_FILE
    with open(filename) as json_file:
        json_data = json.load(json_file)

    filename = NEW_FILE_PATH + f'oos_{oos_n}.json'

    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=3)