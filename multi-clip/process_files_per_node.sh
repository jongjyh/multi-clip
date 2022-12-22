for file in "/home/chenzhongzhi/czz/datasets/la13m_para5m_multilingual/laion2b_multi18lg/total_dataset/$1"_part*.json
do
    bash scripts/run_pred_multinode.sh $file
done