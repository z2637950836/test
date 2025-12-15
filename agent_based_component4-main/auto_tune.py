import os
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product

# ==========================
# 1. 加载 target_data.csv
# ==========================

target_path = "data/target_data.csv"
target_df = pd.read_csv(target_path, header=None, names=["Year", "target_people_number"])
target_df = target_df.set_index("Year")


# ==========================
# 2. 写入 model.props 参数
# ==========================

def modify_model_props(beta0, beta1, rho, cap, windowL):

    with open("props/model.props", "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if "influence.beta0" in line:
            new_lines.append(f"influence.beta0 = {beta0}\n")
        elif "influence.beta1" in line:
            new_lines.append(f"influence.beta1 = {beta1}\n")
        elif "influence.rho" in line:
            new_lines.append(f"influence.rho = {rho}\n")
        elif "influence.cap.per.year" in line:
            new_lines.append(f"influence.cap.per.year = {cap}\n")
        elif "influence.windowL" in line:
            new_lines.append(f"influence.windowL = {windowL}\n")
        else:
            new_lines.append(line)

    with open("props/model.props", "w") as f:
        f.writelines(new_lines)


# ==========================
# 3. 运行模型（无 timeout）
# ==========================

def run_model(beta0, beta1, rho, cap, windowL):

    modify_model_props(beta0, beta1, rho, cap, windowL)

    # 删除旧文件
    if os.path.exists("NumberOfHousehold.csv"):
        os.remove("NumberOfHousehold.csv")

    # 运行模型——不设 timeout
    cmd = ["mpirun", "-n", "1", "bin/main.exe", "props/config.props", "props/model.props"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if proc.returncode != 0:
        print("⚠️ 模型运行失败：returncode =", proc.returncode)
        print(proc.stderr.decode())
        return None, None

    if not os.path.exists("NumberOfHousehold.csv"):
        print("⚠️ 没有生成 NumberOfHousehold.csv")
        return None, None

    sim_df = pd.read_csv("NumberOfHousehold.csv")
    sim_df = sim_df.set_index("Year")

    merged = target_df.join(sim_df, how="inner")
    y_true = merged["target_people_number"].values
    y_pred = merged["Number-of-Households"].values

    mse = np.mean((y_true - y_pred) ** 2)

    return mse, merged


# ==========================
# 4. 网格搜索配置
# ==========================

param_grid = {
    "beta0": [-6, -5, -4],
    "beta1": [5, 8, 10],
    "rho": [0.05, 0.1, 0.2],
    "cap": [0.2, 0.3, 0.4],
    "windowL": [1, 2, 3],
}


# ==========================
# 5. 自动搜索
# ==========================

log_file = f"tune_log_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
best_score = 1e18
best_params = None

with open(log_file, "w") as log:

    for beta0, beta1, rho, cap, windowL in product(
            param_grid["beta0"],
            param_grid["beta1"],
            param_grid["rho"],
            param_grid["cap"],
            param_grid["windowL"]):

        print(f"⭐ 运行：beta0={beta0}, beta1={beta1}, rho={rho}, cap={cap}, windowL={windowL}")
        log.write(f"RUN beta0={beta0}, beta1={beta1}, rho={rho}, cap={cap}, windowL={windowL}\n")

        mse, merged = run_model(beta0, beta1, rho, cap, windowL)

        if mse is None:
            log.write("FAILED\n")
            continue

        log.write(f"MSE={mse}\n")
        print("  → MSE =", mse)

        if mse < best_score:
            best_score = mse
            best_params = (beta0, beta1, rho, cap, windowL)
            merged.to_csv("best_fit_curve.csv")
            log.write("### NEW BEST ###\n")
            print("  🎉 NEW BEST FOUND!")

print("\n==============================")
print("   搜索完成！")
print("   最佳参数：", best_params)
print("   最佳 MSE：", best_score)
print("   最佳结果已保存到 best_fit_curve.csv")
print("==============================")

