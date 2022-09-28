# imports
from utils import *
import shutil
import json
# from julia.api import Julia
# print('Building Julia environment...')
# jpath = '/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia'
# j = Julia(runtime=jpath,compiled_modules=False)

# from julia import Main
# Main.include('sto_imp/compute.jl')
# number of features in the caiso set: 632
# configs
params = {
        #   'n_scenarios':[100],
        #   'n_oos':20,
          'n_scenarios':[100],
          'n_oos':62,
        #   0.81, 14.15, 8.0 seconds 
          'models':['rf']}

hp_search_space_rf = {'n_estimators' : [100],
                   'Xi':[4.0],
                   'max_depth' : [3],
                   'learning_rate' : [1.0],
                   'min_split_loss':['NaN'],
                   'min_samples_split':[6],
                   'max_features':[0.6]
}

hp_search_space_xgb = {'n_estimators' : [50, 100, 150],
                   'Xi':[0.1, 0.25, 1.0, 4.0, 10.0],
                   'max_depth' : [3, 6, 10],
                   'learning_rate' : [0.01, 0.1, 0.3],
                   'min_split_loss':[0.5, 1.0, 2.0],
                   'min_samples_split':['NaN'],
                   'max_features':[0.0477, 0.2, 0.6]
}

hp_search_space_adaboost = {'n_estimators' : [50, 100, 150],
                   'Xi':[0.1, 0.25, 1.0, 4.0, 10.0],
                   'max_depth' : ['NaN'],
                   'learning_rate' : [1.0],
                   'min_split_loss':['NaN'],   
                   'min_samples_split':['NaN'],
                   'max_features':['NaN']
}
hp_search_space_gbt = {'n_estimators' : [50, 100, 150],
                   'Xi':[0.1, 0.25, 1.0, 4.0, 10.0],
                   'max_depth' : [3, 6, 10],
                   'learning_rate' : [0.01, 0.1, 0.3],
                   'min_split_loss':['NaN'],
                   'min_samples_split':[2, 6, 12],
                   'max_features':['NaN']
}

params['rf'] = hp_search_space_rf
params['adaboost'] = hp_search_space_adaboost
params['xgboost'] = hp_search_space_xgb

# paths
INPUT_FILE_DIR = 'input_files/'
OUTPUT_FILE_DIR = 'output_files/'


five_min = False

system = 'caiso'
ts_data = pd.read_csv('data/{}/ts_data_red.csv'.format(system), index_col = 0, parse_dates=['Datetime'])
target_data = pd.read_csv('data/{}/target_data.csv'.format(system), index_col = 0, parse_dates=['Datetime'])
target_data_5min = pd.read_csv('data/{}/target_data_5min.csv'.format(system), index_col = 0, parse_dates=['Datetime'])
if system == 'caiso':
    #max load in the caiso dataset: 41330 MW
    #peak load of the 14-bus system: 321.29 MW
    #total generation capacity of the 14-bus system: 765.31 MW
    load_scal = 765.31*0.90/41330
else:
    #max load in the nyiso dataset: 31866.4167 MW on Aug 29, 2018 at 5 pm
    load_scal = 765.31*0.90/31866.74
REF_FILE = 'data/{}.json'.format(system)
if five_min == True:
    REF_FILE_OOS = 'data/{}_5min.json'.format(system)
else:
    REF_FILE_OOS = 'data/{}.json'.format(system)
finished_conf_f = 'experiments/finished_experiments.csv'
for n_s in params['n_scenarios']:
    NEW_FILE_PATH = f'input_files/snum_{n_s}/'
    os.makedirs(NEW_FILE_PATH)
    for n in range(1,params['n_oos']+1):
        NEW_FILE_PATH = f'input_files/snum_{n_s}/oos_{n}/'
        os.makedirs(NEW_FILE_PATH)
        create_oos_files(REF_FILE_OOS, NEW_FILE_PATH, n)
        NEW_FILE_PATH = f'input_files/snum_{n_s}/oos_{n}/naive/naive/'
        os.makedirs(NEW_FILE_PATH)
        create_naive_files(REF_FILE, NEW_FILE_PATH, n_s)
        for m in params['models']:
            NEW_FILE_PATH = f'input_files/snum_{n_s}/oos_{n}/weighted/{m}/'
            os.makedirs(NEW_FILE_PATH)
            create_new_scenario_files(REF_FILE, NEW_FILE_PATH, m, n_s, params[m])
            NEW_FILE_PATH = f'input_files/snum_{n_s}/oos_{n}/point/{m}/'
            os.makedirs(NEW_FILE_PATH)
            create_new_scenario_files_point(REF_FILE, NEW_FILE_PATH, m, 1, params[m])
finished_conf_f = 'experiments/finished_experiments.csv'
# for n_s in params['n_scenarios']:
#     NEW_FILE_PATH = f'input_files/snum_{n_s}/'
#     os.makedirs(NEW_FILE_PATH)
#     for n in range(1,params['n_oos']+1):
#         NEW_FILE_PATH = f'input_files/snum_{n_s}/oos_{n}/'
#         os.makedirs(NEW_FILE_PATH)
#         create_oos_files(REF_FILE_OOS, NEW_FILE_PATH, n)
#         NEW_FILE_PATH = f'input_files/snum_{n_s}/oos_{n}/naive/naive/'
#         os.makedirs(NEW_FILE_PATH)
#         create_naive_files(REF_FILE, NEW_FILE_PATH, n_s)
#         for m in params['models']:
#             NEW_FILE_PATH = f'input_files/snum_{n_s}/oos_{n}/weighted/{m}/'
#             os.makedirs(NEW_FILE_PATH)
#             create_new_scenario_files(REF_FILE, NEW_FILE_PATH, m, n_s, params[m])
#             NEW_FILE_PATH = f'input_files/snum_{n_s}/oos_{n}/point/{m}/'
#             os.makedirs(NEW_FILE_PATH)
#             create_new_scenario_files_point(REF_FILE, NEW_FILE_PATH, m, 1, params[m])

for m in params['models']:    
    for n_s in params['n_scenarios']:
        hp_search_space, n_oos = assign_params(m, params)
        print(f"\n=== starting experiment with {n_s} scenarios ===")
        X_train, X_test, y_train, y_test, X_scaler, y_scaler = prepare_forecast_data(ts_data, target_data, n_scenarios=n_s, load_scaling = load_scal, n_oos=n_oos, start_date_train='2018-06-01', start_date_oos='2019-07-01',)
        if five_min == True:
            _, _, y_train_5min, y_test_5min, _, y_scaler_5min = prepare_forecast_data(ts_data, target_data_5min, n_scenarios=n_s, load_scaling = load_scal, n_oos=n_oos, start_date_train='2018-06-01', start_date_oos='2019-07-01',)
        for n in range(1,n_oos+1):
            filename = INPUT_FILE_DIR + f'snum_{str(n_s)}/oos_{str(n)}/naive/naive/naive.json'
            with open(filename) as json_file:
                json_data = json.load(json_file)
            write_naive_scenario_loads_to_json(json_data, y_train, filename,y_scaler, n_s)
            filename = INPUT_FILE_DIR + f'snum_{str(n_s)}/oos_{str(n)}/oos_{str(n)}.json'
            with open(filename) as json_file:
                json_data = json.load(json_file)
            if five_min == True:
                write_5min_oos_loads_to_json(json_data,y_test_5min,n,filename,y_scaler_5min)
            else:
                write_oos_loads_to_json(json_data,y_test,n,filename,y_scaler)
 
        for n_estimators in hp_search_space['n_estimators']:
            for max_depth in hp_search_space['max_depth']:
                for learning_rate in hp_search_space['learning_rate']:
                    for min_split_loss in hp_search_space['min_split_loss']:
                        for min_samples_split in hp_search_space['min_samples_split']:
                            for max_features in hp_search_space['max_features']:
                                hp = {'n_estimators' : n_estimators,
                                        'Xi' : 'xi',
                                        'max_depth' : max_depth,
                                        'learning_rate': learning_rate,
                                        'min_split_loss': min_split_loss,
                                        'min_samples_split': min_samples_split,
                                        'max_features': max_features}
                                    
                                run_hp_experiment_for_point_prediction(m, hp, X_train, X_test, y_train,n_oos,n_s,y_scaler)
                                for xi in hp_search_space['Xi']:
                                    hp = {'n_estimators' : n_estimators,
                                        'Xi' : xi,
                                        'max_depth' : max_depth,
                                        'learning_rate': learning_rate,
                                        'min_split_loss': min_split_loss,
                                        'min_samples_split': min_samples_split,
                                        'max_features': max_features}
                                    run_hp_experiment(m, hp, X_train, X_test, y_train,n_oos,n_s,y_scaler)
                                    
                
        conf = [m,n_s]
        with open(finished_conf_f,'a') as f:
            wr = csv.writer(f)
            wr.writerow(conf)
                