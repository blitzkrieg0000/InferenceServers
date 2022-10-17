docker run --rm --runtime=nvidia -it \
    -v /srv/nfs/exports/InferenceServerModels/TFServing/models:/tmp tensorflow/tensorflow:latest-gpu \
    /usr/local/bin/saved_model_cli convert \
    --dir /tmp/tracknet/1 \
    --output_dir /tmp/tracknet_trt/1 \
    --tag_set serve \
    tensorrt --precision_mode FP32
    
    
    
    --max_batch_size 1 \
    --is_dynamic_op True