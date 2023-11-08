# smile-face-simulation

## conditional training

train pipeline based on the pretrained decoder 

    python scripts/train_conditional_network.py \
    --decoder_checkpoint_path /mnt/share/shenfeihong/weight/smile/cond_decoder.pt \
    --data_path /mnt/share/shenfeihong/weight/smile/cond_data
