#!/bin/bash
cd "/c/Users/t-bvangala/Downloads/SC26_AgenticAI4HPC"
export RESULTS_FILE="results/pareval_scaled.jsonl"
export VERIFY_BACKEND="vm"
MODELS="gpt-5.4 gpt-5.2 gpt-5.3-codex gpt-5.4-pro"
TASKS="20_histogram_pixel_histogram 25_reduce_xor 30_scan_prefix_sum 15_graph_edge_count 36_search_check_if_array_contains_value 40_sort_sort_an_array_of_complex_numbers_by_magnitude 45_sparse_la_sparse_solve 00_dense_la_lu_decomp 05_fft_inverse_fft 10_geometry_convex_hull 50_stencil_xor_kernel 55_transform_relu 17_graph_highest_degree 21_histogram_bin_0-100 22_histogram_count_quadrants 26_reduce_product_of_inverses 29_reduce_sum_of_min_of_pairs 33_scan_reverse_prefix_sum 35_search_search_for_last_struct_by_key 38_search_find_the_first_even_number"
i=0; n=$(echo $TASKS | wc -w)
for t in $TASKS; do
  i=$((i+1))
  echo "[$(date +%H:%M:%S)] ($i/$n) TASK $t : generating+scoring 4 models x k=8 ..."
  python -m harness.run_pareval_batch --models $MODELS --tasks "$t" --conditions single_shot --trials 8 2>&1 | tail -3
  echo "[$(date +%H:%M:%S)] ($i/$n) $t done; scaled pool now $(wc -l < results/pareval_scaled.jsonl) rows"
done
echo "ALL SCALED TASKS COMPLETE: $(wc -l < results/pareval_scaled.jsonl) rows in pareval_scaled.jsonl"
