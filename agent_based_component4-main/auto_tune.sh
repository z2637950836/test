#!/bin/bash

# ======================================================
#  Auto Parameter Tuning (MAE / MSE fit to target curve)
#  Fully Unix LF Script
# ======================================================

set -e

TARGET_FILE="data/target_data.csv"

if [[ ! -f "$TARGET_FILE" ]]; then
    echo "❌ ERROR: data/target_data.csv not found"
    exit 1
fi

mkdir -p outputs/configs
mkdir -p outputs/logs
mkdir -p outputs/raw_results

# -------- Load target curve into dictionary --------
declare -A TARGET_DICT
while IFS=',' read -r yr val; do
    TARGET_DICT[$yr]=$val
done < "$TARGET_FILE"

echo "Loaded ${#TARGET_DICT[@]} target points."

BASE_CONFIG="props/model.props"
BASE_COPY="outputs/configs/base_model.props"
cp "$BASE_CONFIG" "$BASE_COPY"

# -------- Parameter search ranges --------
WINDOWS=(1 2 3)
RHOS=(0.05 0.1 0.2)
BETA0S=(-6 -5 -4)
BETA1S=(4 5 6)
CAPS=(0.10 0.20 0.30)

RESULTS="outputs/tune_results.csv"
echo "windowL,rho,beta0,beta1,cap,MAE,MSE" > "$RESULTS"

# -------- Function: compute MAE / MSE --------
compute_error() {
    local sim="$1"

    local mae_sum=0
    local mse_sum=0
    local count=0

    while IFS=',' read -r yr sim_val; do
        [[ "$yr" == "Year" ]] && continue

        tgt_val=${TARGET_DICT[$yr]}
        if [[ "$tgt_val" != "" ]]; then
            diff=$(echo "$sim_val - $tgt_val" | bc -l)
            abs_diff=$(echo "${diff#-}" | bc -l)
            sq_diff=$(echo "$diff * $diff" | bc -l)

            mae_sum=$(echo "$mae_sum + $abs_diff" | bc -l)
            mse_sum=$(echo "$mse_sum + $sq_diff" | bc -l)
            count=$((count + 1))
        fi
    done < "$sim"

    if [[ $count -eq 0 ]]; then
        echo "9999,9999"
    else
        mae=$(echo "$mae_sum / $count" | bc -l)
        mse=$(echo "$mse_sum / $count" | bc -l)
        echo "$mae,$mse"
    fi
}

# -------- Function: run model --------
run_model() {
    local cfg="$1"
    local log="$2"

    mpirun -bind-to none -n 1 bin/main.exe props/config.props "$cfg" > "$log" 2>&1
}

# ======================================================
#                MAIN GRID SEARCH LOOP
# ======================================================

BEST_MSE=999999
BEST_LINE=""

for L in "${WINDOWS[@]}"; do
for rho in "${RHOS[@]}"; do
for b0 in "${BETA0S[@]}"; do
for b1 in "${BETA1S[@]}"; do
for cap in "${CAPS[@]}"; do

    name="L${L}_rho${rho}_b0${b0}_b1${b1}_cap${cap}"
    cfg="outputs/configs/${name}.props"
    out="outputs/raw_results/${name}.csv"
    log="outputs/logs/${name}.log"

    cp "$BASE_COPY" "$cfg"

    # Rewrite influence parameters
    awk -v L="$L" -v rho="$rho" -v b0="$b0" -v b1="$b1" -v cap="$cap" -v out="$out" '
    {
        if ($1=="influence.windowL") print "influence.windowL = " L;
        else if ($1=="influence.rho") print "influence.rho = " rho;
        else if ($1=="influence.beta0") print "influence.beta0 = " b0;
        else if ($1=="influence.beta1") print "influence.beta1 = " b1;
        else if ($1=="influence.cap.per.year") print "influence.cap.per.year = " cap;
        else if ($1=="result.file") print "result.file = " out;
        else print $0;
    }' "$cfg" > "${cfg}.tmp" && mv "${cfg}.tmp" "$cfg"

    echo "▶ Running $name ..."
    run_model "$cfg" "$log"

    ERR=$(compute_error "$out")
    MAE=$(echo "$ERR" | cut -d',' -f1)
    MSE=$(echo "$ERR" | cut -d',' -f2)

    echo "$L,$rho,$b0,$b1,$cap,$MAE,$MSE" >> "$RESULTS"

    smaller=$(echo "$MSE < $BEST_MSE" | bc -l)
    if [[ "$smaller" -eq 1 ]]; then
        BEST_MSE=$MSE
        BEST_LINE="$L,$rho,$b0,$b1,$cap,$MAE,$MSE"
    fi

done
done
done
done
done

echo "============================================"
echo " BEST PARAMETER SET (by MSE):"
echo "   $BEST_LINE"
echo "============================================"

rm "$BASE_COPY"

echo "All results saved → outputs/tune_results.csv"

