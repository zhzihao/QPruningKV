export CUDA_VISIBLE_DEVICES=$1

method=LayerQuant # Support PyramidKV, SnapKV, H2O, StreamingLLM,LayerQuant
max_capacity_prompts=128 # 128,2048 in paper
attn_implementation=flash_attention_2 
result_path=./
model_path=$2 # path to model
quant_method=layerquant # Support kivi layerquant and kvquant, default None.
nbits=4 # Quantization bit-width support 8,4,2. Need to set quant_method first.
save_dir=${result_path}"results_ruler" # path to result save_dir
quant_layers_bits=8 
quant_begin_layers=0
quant_end_layers=4

python3 run_ruler.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --use_cache True \
    --nbits ${nbits} \
    --quant_method ${quant_method} \
    --quant_layers_bits ${quant_layers_bits} \
    --quant_begin_layers ${quant_begin_layers} \
    --quant_end_layers ${quant_end_layers}
