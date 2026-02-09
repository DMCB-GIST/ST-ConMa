# HEG (Highly Expressed Genes)
for i in {0..7}; do python ./evaluations/gep_pred/train_st_conma.py --dataset her2st --fold $i --gene_type heg --use_igm --output_dir ./results/gep_pred/st_conma_igc_igm_noPT --top_k none --device cuda:1 --checkpoint none --epochs 12; done
for i in {0..3}; do python ./evaluations/gep_pred/train_st_conma.py --dataset cscc --fold $i --gene_type heg --use_igm --output_dir ./results/gep_pred/st_conma_igc_igm_noPT --top_k none --device cuda:1 --checkpoint none --epochs 12; done
for i in {0..3}; do python ./evaluations/gep_pred/train_st_conma.py --dataset hlt --fold $i --gene_type heg --use_igm --output_dir ./results/gep_pred/st_conma_igc_igm_noPT --top_k none --device cuda:1 --checkpoint none --epochs 12; done

# HVG (Highly Variable Genes)
for i in {0..7}; do python ./evaluations/gep_pred/train_st_conma.py --dataset her2st --fold $i --gene_type hvg --use_igm --output_dir ./results/gep_pred/st_conma_igc_igm_noPT --top_k none --device cuda:1 --checkpoint none --epochs 12; done
for i in {0..3}; do python ./evaluations/gep_pred/train_st_conma.py --dataset cscc --fold $i --gene_type hvg --use_igm --output_dir ./results/gep_pred/st_conma_igc_igm_noPT --top_k none --device cuda:1 --checkpoint none --epochs 12; done
for i in {0..3}; do python ./evaluations/gep_pred/train_st_conma.py --dataset hlt --fold $i --gene_type hvg --use_igm --output_dir ./results/gep_pred/st_conma_igc_igm_noPT --top_k none --device cuda:1 --checkpoint none --epochs 12; done

echo "Done"
