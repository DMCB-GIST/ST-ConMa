# HEG (Highly Expressed Genes)
for i in {0..7}; do python ./evaluations/gep_pred/eval_st_conma_zeroshot.py --dataset her2st --fold $i --gene_type heg; done
for i in {0..3}; do python ./evaluations/gep_pred/eval_st_conma_zeroshot.py --dataset cscc --fold $i --gene_type heg; done
for i in {0..3}; do python ./evaluations/gep_pred/eval_st_conma_zeroshot.py --dataset hlt --fold $i --gene_type heg; done

# HVG (Highly Variable Genes)
for i in {0..7}; do python ./evaluations/gep_pred/eval_st_conma_zeroshot.py --dataset her2st --fold $i --gene_type hvg; done
for i in {0..3}; do python ./evaluations/gep_pred/eval_st_conma_zeroshot.py --dataset cscc --fold $i --gene_type hvg; done
for i in {0..3}; do python ./evaluations/gep_pred/eval_st_conma_zeroshot.py --dataset hlt --fold $i --gene_type hvg; done

echo "Done"