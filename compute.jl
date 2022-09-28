using Base: Float64, @var, current_logger, Ordered
using Gurobi
using Cbc
using Clp
using JuMP
using JSON
using PyCall
using Printf
import Base
using JSON: print
using GZip
import Base: getindex, time
using DataStructures
import MathOptInterface
using LinearAlgebra

basedir = homedir()
push!(LOAD_PATH,"$basedir/flexibility_repo/cso/6AWag/src/")
imp_dir = "$basedir/flexibility_repo/cso"
using UnitCommitmentSTO
import MathOptInterface
using LinearAlgebra
import UnitCommitmentSTO:
    Formulation,
    KnuOstWat2018,
    MorLatRam2013,
    ShiftFactorsFormulation
function val_mod(x)
    if abs(value(x))<1e-4
        return  0.0
    end
    return round(abs(value(x)), digits=6)
end    

function compute_total_payment(model, conf_path, gen_pay, lmp_vals, curt_vals)
    vals = Dict(v=> val_mod(v) for v in all_variables(model) if is_binary(v))
    for (v, val) in vals
        fix(v, val)
    end
    relax_integrality(model)
    JuMP.optimize!(model)
    instance = model[:instance]
    lmp_vals_temp=OrderedDict() 
    gen_pay_temp=OrderedDict() 
    curt_vals_temp=OrderedDict()
    for bus in instance.buses
        for t in 1:instance.time
            push!(lmp_vals_temp, ["$(bus.name), $(t)"]=>-shadow_price(model[:eq_net_injection_def]["s1", bus.name, t]) * instance.time_multiplier)
        end
    end
    push!(lmp_vals, conf_path => lmp_vals_temp )
    for g in instance.units
        for t in 1:instance.time
            push!(gen_pay_temp, ["$(g.name), $(t)"]=> value(model[:prod_above]["s1", g.name, t] +
                    (model[:is_on][g.name, t] * g.min_power[t])) * 
                    (-shadow_price(model[:eq_net_injection_def]["s1", g.bus.name, t]))  )
        end
    end
    push!(gen_pay, conf_path => gen_pay_temp )
    for bus in instance.buses
        for t in 1:instance.time
            push!(curt_vals_temp, [bus.name, t]=>value(model[:curtail]["s1", bus.name,t]) / instance.time_multiplier)
        end
    end
    push!(curt_vals, conf_path => curt_vals_temp )

end
function first_stage(model_type, scen_num, oos_n, conf, conf_path, type)
    
    instance = UnitCommitmentSTO.read("$imp_dir/input_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/$(conf_path).json",)
    model = UnitCommitmentSTO.build_model(
        instance = instance,
        optimizer = Gurobi.Optimizer,
        formulation = Formulation(
            
            )
    )
    time_stat = @timed UnitCommitmentSTO.optimize!(model)
    solution = UnitCommitmentSTO.solution(model)
    UnitCommitmentSTO.write("$imp_dir/input_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/$(conf_path)_fs.json", solution)
    return time_stat
end

function second_stage(model_type, scen_num, oos_n, conf, conf_path, type, gen_pay, lmp_vals, obj_vals, curt_vals)
    
    instance = UnitCommitmentSTO.read("$imp_dir/input_files/snum_$(scen_num)/oos_$(oos_n)/oos_$(oos_n).json",)
    model = UnitCommitmentSTO.build_model(
    instance = instance,
    optimizer = Gurobi.Optimizer,
    formulation = Formulation(
        )
    )
    mul = instance.time_multiplier
    
    if type !== "ideal"
        inp_file = "$imp_dir/input_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/$(conf_path)_fs.json"
        inp_json = JSON.parse(open(inp_file, "r"), dicttype = () -> DefaultOrderedDict(nothing))
        [@constraint(model, model[:is_on][g.name,t] == inp_json["Is on"][g.name][div(t-1,mul)+1]) for g in instance.units for t in 1:instance.time]
    end
    UnitCommitmentSTO.optimize!(model)
    solution = UnitCommitmentSTO.solution(model)
    UnitCommitmentSTO.write("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/$(conf_path)_sol.json", solution)
    
    push!(obj_vals, conf_path => objective_value(model))
    compute_total_payment(model, conf_path, gen_pay, lmp_vals, curt_vals)

end



function retrieve_best_hyperparameters(imp_dir, scen_num, params, types)
    json_f = open("$imp_dir/output_files/snum_$(scen_num)/cost_results.json", "r")
    json_fs=JSON.parse(json_f, dicttype = () -> DefaultOrderedDict(nothing))
    for type in types
        if type === "weighted" || type === "point"
            model_names = params["models"]
        else
            model_names = [type]
        end
        for model_type in model_names
            min_key = reduce((x, y) -> json_fs[type][model_type]["configuration costs"][x]["sum"] ≤ json_fs[type][model_type]["configuration costs"][y]["sum"] ? x : y, keys(json_fs[type][model_type]["configuration costs"]))
            min_val = json_fs[type][model_type]["configuration costs"][min_key]["sum"]
            temp = OrderedDict(min_key => min_val)
            json_fs[type][model_type]["best configuration"] = temp
        end
    end
    open("$imp_dir/output_files/snum_$(scen_num)/cost_results.json","w") do f
        JSON.print(f, json_fs)
    end
    json_f = open("$imp_dir/output_files/snum_$(scen_num)/payment_results.json", "r")
    json_fs=JSON.parse(json_f, dicttype = () -> DefaultOrderedDict(nothing))
    for type in types
        if type === "weighted" || type === "point"
            model_names = params["models"]
        else
            model_names = [type]
        end
        for model_type in model_names
            min_key = reduce((x, y) -> json_fs[type][model_type]["configuration payments"][x]["sum"] ≤ json_fs[type][model_type]["configuration payments"][y]["sum"] ? x : y, keys(json_fs[type][model_type]["configuration payments"]))
            min_val = json_fs[type][model_type]["configuration payments"][min_key]["sum"]
            temp = OrderedDict(min_key => min_val)
            json_fs[type][model_type]["best configuration"] = temp
        end
    end
    open("$imp_dir/output_files/snum_$(scen_num)/payment_results.json","w") do f
        JSON.print(f, json_fs)
    end

end

function write_results(imp_dir, scen_num, params, types)
    payment_results = OrderedDict()
    cost_results = OrderedDict()
    curt_results = OrderedDict()
    lmp_results=OrderedDict()
    for type in types
        if type === "weighted" || type === "point"
            model_names = params["models"]
        else
            model_names = [type]
        end
        type_oos_payment = OrderedDict()
        type_oos_cost = OrderedDict()
        type_oos_curt = OrderedDict()
        type_oos_lmp = OrderedDict()
        for model_type in model_names
            model_oos_payment = OrderedDict()
            temp = OrderedDict()
            model_oos_cost = OrderedDict()
            temp_cost = OrderedDict()
            model_oos_curt = OrderedDict()
            temp_curt = OrderedDict()
            model_oos_lmp = OrderedDict()
            temp_lmp = OrderedDict()
            for oos_n in 1:params["n_oos"]
                oos_payments = OrderedDict()
                json_f = open("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/gen_payment.json", "r")
                dict=JSON.parse(json_f, dicttype = () -> DefaultOrderedDict(nothing))
                
                for (key, value) in dict["total payments"]
                    push!(oos_payments, key => value)
                end
                push!(temp, oos_n=> oos_payments)
                oos_lmps = OrderedDict()
                json_f = open("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/lmp_values.json", "r")
                dict=JSON.parse(json_f, dicttype = () -> DefaultOrderedDict(nothing))
                
                for (key, value) in dict
                    push!(oos_lmps, key => value)
                end
                push!(temp_lmp, oos_n=> oos_lmps)
                oos_costs = OrderedDict()
                json_f_costs = open("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/obj_vals.json", "r")
                dict_costs=JSON.parse(json_f_costs, dicttype = () -> DefaultOrderedDict(nothing))
                
                for (key, value) in dict_costs
                    push!(oos_costs, key => value)
                end
                push!(temp_cost, oos_n=> oos_costs)
                oos_curt = OrderedDict()
                json_f_curt = open("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/curt_vals.json", "r")
                dict_curt=JSON.parse(json_f_curt, dicttype = () -> DefaultOrderedDict(nothing))
                
                for (key, value) in dict_curt["total curtailment"]
                    push!(oos_curt, key => value)
                end
                push!(temp_curt, oos_n=> oos_curt)
            end
            push!(model_oos_payment, "oos payments"=> temp)
            push!(model_oos_cost, "oos costs"=> temp_cost)
            push!(model_oos_curt, "oos curts"=> temp_curt)
            push!(model_oos_lmp, "oos lmps"=> temp_lmp)
            json_f = open("$imp_dir/output_files/snum_$(scen_num)/oos_1/$(type)/$(model_type)/gen_payment.json", "r")
            dict=JSON.parse(json_f, dicttype = () -> DefaultOrderedDict(nothing))
            temp = OrderedDict()
            temp_cost = OrderedDict()
            temp_curt = OrderedDict()
            temp_lmp = OrderedDict()
            for (key, value) in dict["total payments"]
                temp2 = OrderedDict()
                temp2_cost = OrderedDict()
                temp2_curt = OrderedDict()
                temp2_temp_lmp = OrderedDict()
                for i in 1:params["n_oos"]
                    push!(temp2, i => model_oos_payment["oos payments"][i][key])
                    push!(temp2_cost, i => model_oos_cost["oos costs"][i][key])
                    push!(temp2_curt, i => model_oos_curt["oos curts"][i][key])
                    push!(temp2_temp_lmp, i => model_oos_lmp["oos lmps"][i][key])
                end
                
                push!(temp2,"sum" =>Base.sum([model_oos_payment["oos payments"][i][key] for i in 1:params["n_oos"]])) 
                push!(temp, key=>temp2)

                push!(temp2_cost,"sum" =>Base.sum([model_oos_cost["oos costs"][i][key] for i in 1:params["n_oos"]])) 
                push!(temp_cost, key=>temp2_cost)

                push!(temp2_curt,"sum" =>Base.sum([model_oos_curt["oos curts"][i][key] for i in 1:params["n_oos"]])) 
                push!(temp_curt, key=>temp2_curt)
                temp2_lmp = OrderedDict()
                for (key2, val) in model_oos_lmp["oos lmps"][1][key]
                    push!(temp2_lmp, key2 =>Base.sum([model_oos_lmp["oos lmps"][i][key][key2] for i in 1:params["n_oos"]])/params["n_oos"]) 
                end
                
                push!(temp2_temp_lmp, "average"=>temp2_lmp)
                push!(temp_lmp, key=>temp2_temp_lmp)
                
            end
            push!(model_oos_payment, "configuration payments"=>temp)
            push!(type_oos_payment, model_type => model_oos_payment)
            push!(model_oos_cost, "configuration costs"=>temp_cost)
            push!(type_oos_cost, model_type => model_oos_cost)
            push!(model_oos_curt, "configuration curts"=>temp_curt)
            push!(type_oos_curt, model_type => model_oos_curt)
            push!(model_oos_lmp, "configuration lmps"=>temp_lmp)
            push!(type_oos_lmp, model_type => model_oos_lmp)
        end
        push!(payment_results, type => type_oos_payment)
        push!(cost_results, type => type_oos_cost)
        push!(curt_results, type => type_oos_curt)
        push!(lmp_results, type => type_oos_lmp)
    end
  
    open("$imp_dir/output_files/snum_$(scen_num)/payment_results.json","w") do f
        JSON.print(f, payment_results)
    end
    open("$imp_dir/output_files/snum_$(scen_num)/cost_results.json","w") do f
        JSON.print(f, cost_results)
    end
    open("$imp_dir/output_files/snum_$(scen_num)/curt_results.json","w") do f
        JSON.print(f, curt_results)
    end
    open("$imp_dir/output_files/snum_$(scen_num)/lmp_results.json","w") do f
        JSON.print(f, lmp_results)
    end
end

function mainfunc()

    params = Dict("n_scenarios"=>[100],
          "n_oos"=>62,
          "models"=>["rf"]
    )

    hp_search_space_rf = Dict("n_estimators" => [100],
                    "Xi" => [4.0],
                    "max_depth" => [3],
                    "learning_rate" => [1.0],
                    "min_split_loss"=>["NaN"],
                    "min_samples_split"=>[6],
                    "max_features"=>[0.6]
    )

 

    params["rf"] = hp_search_space_rf

    types = ["weighted"]
    mkdir("$imp_dir/output_files/")
    stat_dict = Dict()
    for scen_num in params["n_scenarios"]
        for type in types
            for oos_n in 1:params["n_oos"]
                if type === "weighted" || type === "point"
                    filename = "$imp_dir/output_files/$(scen_num)/oos_$(oos_n)/oos_$(oos_n)_results.json"
                    mkpath("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/")
                    for model_type in params["models"]
                        mkdir("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/")
                        
                        gen_pay_det = OrderedDict()
                        obj_vals = OrderedDict() 
                        lmp_vals = OrderedDict()  
                        curt_vals_det = OrderedDict()
                      
                        for n_est in params[model_type]["n_estimators"]
                            for m_dep in params[model_type]["max_depth"]
                                for l_rat in params[model_type]["learning_rate"]
                                    for m_spl in params[model_type]["min_split_loss"]
                                        for m_sam in params[model_type]["min_samples_split"]
                                            for m_fea in params[model_type]["max_features"]
                                                if type === "point"
                                                    conf = Dict("n_estimators" => n_est, "Xi" => "xi", "max_depth" => m_dep, "learning_rate" => l_rat, "min_split_loss" => m_spl, "min_samples_split" => m_sam, "max_features" => m_fea )
                                                    conf_path = "n_est_$(n_est)_Xi_xi_max_depth_$(m_dep)_lear_rate_$(l_rat)_min_sp_l_$(m_spl)_min_samples_split_$(m_sam)_max_features_$(m_fea)"
                                                    time_val = first_stage(model_type, scen_num, oos_n, conf, conf_path, type)
                                                    second_stage(model_type, scen_num, oos_n, conf, conf_path, type, gen_pay_det, lmp_vals, obj_vals, curt_vals_det)
                                                    stat_key = "point_model_$(model_type)_snum_$(scen_num)_n_est_$(n_est)_max_depth_$(m_dep)_lear_rate_$(l_rat)_min_sp_l_$(m_spl)_min_samples_split_$(m_sam)_max_features_$(m_fea)_oos_$(oos_n)"
                                                    stat_dict[stat_key] = time_val
                                                else
                                                    for xi in params[model_type]["Xi"]
                                                       
                                                        conf = Dict("n_estimators" => n_est, "Xi" => xi, "max_depth" => m_dep, "learning_rate" => l_rat, "min_split_loss" => m_spl, "min_samples_split" => m_sam, "max_features" => m_fea )
                                                        conf_path = "n_est_$(n_est)_Xi_$(xi)_max_depth_$(m_dep)_lear_rate_$(l_rat)_min_sp_l_$(m_spl)_min_samples_split_$(m_sam)_max_features_$(m_fea)"
                                                        
                                                        time_val = first_stage(model_type, scen_num, oos_n, conf, conf_path, type)
                                                        second_stage(model_type, scen_num, oos_n, conf, conf_path, type, gen_pay_det, lmp_vals, obj_vals, curt_vals_det)
                                                        stat_key = "weighted_model_$(model_type)_snum_$(scen_num)_n_est_$(n_est)_Xi_$(xi)_max_depth_$(m_dep)_lear_rate_$(l_rat)_min_sp_l_$(m_spl)_min_samples_split_$(m_sam)_max_features_$(m_fea)_oos_$(oos_n)"
                                                        stat_dict[stat_key] = time_val
                                                        
                                                    end
                                                end

                                            end
                                        end
                                    end
                                end
                            end
                        end

                        gen_pay = OrderedDict()
                        push!(gen_pay, "detailed payment"=>gen_pay_det)
                        temp = OrderedDict()
                        for (key, dict) in gen_pay_det
                            push!(temp, key => sum([val for (key2, val) in dict]))
                        end
                        push!(gen_pay, "total payments"=>temp)
                        open("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/obj_vals.json","w") do f
                            JSON.print(f, obj_vals)
                        end
                        curt_vals = OrderedDict()
                        push!(curt_vals, "detailed curtailment"=>curt_vals_det)
                        temp_curt = OrderedDict()
                        for (key, dict) in curt_vals_det
                            push!(temp_curt, key => sum([val for (key2, val) in dict]))
                        end
                        push!(curt_vals, "total curtailment"=>temp_curt)
                        open("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/curt_vals.json","w") do f
                            JSON.print(f, curt_vals)
                        end
                        open("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/gen_payment.json","w") do f
                            JSON.print(f, gen_pay)
                        end
                        open("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/lmp_values.json","w") do f
                            JSON.print(f, lmp_vals)
                        end

                    end
                else
                    gen_pay_det = OrderedDict() 
                    lmp_vals = OrderedDict()  
                    obj_vals = OrderedDict() 
                    curt_vals_det = OrderedDict() 
                    conf = conf_path = model_type = type
                    mkpath("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)")

                    if type === "naive"
                        time_val = first_stage(model_type, scen_num, oos_n, conf, conf_path, type)
                        stat_key = "naive_snum_$(scen_num)_oos_$(oos_n)"
                        stat_dict[stat_key] = time_val
                    end
                    second_stage(model_type, scen_num, oos_n, conf, conf_path, type, gen_pay_det, lmp_vals, obj_vals, curt_vals_det)
                    
                    open("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/obj_vals.json","w") do f
                        JSON.print(f, obj_vals)
                    end
                    curt_vals = OrderedDict()
                    push!(curt_vals, "detailed curtailment"=>curt_vals_det)
                    temp_curt = OrderedDict()
                    for (key, dict) in curt_vals_det
                        push!(temp_curt, key => sum([val for (key2, val) in dict]))
                    end
                    push!(curt_vals, "total curtailment"=>temp_curt)
                    open("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/curt_vals.json","w") do f
                        JSON.print(f, curt_vals)
                    end
                    gen_pay = OrderedDict()
                    push!(gen_pay, "detailed payment"=>gen_pay_det)
                    temp = OrderedDict()
                    for (key, dict) in gen_pay_det
                        push!(temp, key => sum([val for (key2, val) in dict]))
                    end
                    push!(gen_pay, "total payments"=>temp)

                    open("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/gen_payment.json","w") do f
                        JSON.print(f, gen_pay)
                    end
                    open("$imp_dir/output_files/snum_$(scen_num)/oos_$(oos_n)/$(type)/$(model_type)/lmp_values.json","w") do f
                        JSON.print(f, lmp_vals)
                    end
                end
            end
        end 
        write_results(imp_dir, scen_num, params, types)
        retrieve_best_hyperparameters(imp_dir, scen_num, params, types)
    end
    return stat_dict
end

stat_dict = mainfunc()
open("$imp_dir/output_files/stats.json","w") do f
    JSON.print(f, stat_dict)     
end               