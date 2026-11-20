DRY_RUN=1 OPT=muon CONFIG=llama3_r1 bash ladder/sweep.sh                    # 查看会提交的任务
OPT=muon CONFIG=llama3_r1 LRS=1e-3 bash ladder/sweep.sh                     # 第一个 LR
# 等日志出现 "after dataloaders are built"（数据集索引已建好）再提交剩下 4 个
OPT=muon CONFIG=llama3_r1 LRS="5e-4 7e-4 1.4e-3 2e-3" bash ladder/sweep.sh
