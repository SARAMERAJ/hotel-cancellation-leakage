#!/bin/bash
cd "$(dirname "$0")"
done_(){ tail -1 "out/$1.log" 2>/dev/null | grep -q DONE; }
cv(){ n=$1; shift; done_ $n || python3 run_cv.py $n "$@" > /dev/null 2>&1; }
tm(){ n=$1; shift; done_ $n || python3 run_time.py $n "$@" > /dev/null 2>&1; }
ex(){ done_ $1 || python3 run_extra.py $1 > out/$1.err 2>&1; }
G4="Logistic Regression,XGBoost,LightGBM,DNN"
F="Logistic Regression,XGBoost,LightGBM"
mkdir -p out
cv base_gb --models "$F"
cv noparking_gb --drop required_car_parking_spaces --models "$F"
cv rawrandom_gb --raw --models "$F"
cv rawgroup_gb --raw --group --models "$F"
cv globalenc_gb --global-enc --models "$F"
cv smotenc_gb --resample smotenc --models "$F"
cv cw_gb --resample none --class-weight --models "$F"
tm timearr_gb --mode arrival --models "$F"
cv base_dnn --models DNN
cv noparking_dnn --drop required_car_parking_spaces --models DNN
cv rawrandom_dnn --raw --models DNN
cv rawgroup_dnn --raw --group --models DNN
cv base_cnnh --models "1D CNN,CNN+DNN Hybrid"
cv noparking_cnnh --drop required_car_parking_spaces --models "1D CNN,CNN+DNN Hybrid"
tm timearr_dnn --mode arrival --models DNN
tm timebook_g4 --mode booking --models "$G4"
cv postbook_g4 --drop required_car_parking_spaces,total_of_special_requests --models "$G4"
cv smotenc_dnn --resample smotenc --models DNN
cv cw_dnn --resample none --class-weight --models DNN
cv globalenc_dnn --global-enc --models DNN
ex eqbudget
ex ftt
ex tabnet
tm timearr_cnnh --mode arrival --models "1D CNN,CNN+DNN Hybrid"
python3 drift.py > out/drift.out 2>&1
python3 analyze.py
