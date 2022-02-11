# Run

To reproduce the LSTM experiment, run
```bash
python3 -m benchmarks --name lstm_crowd_embeddings \
         --dataset=imdb \
         --train_path=data/imdb/train_crowd_alpha047.csv  --val_path=data/imdb/val_clean_alpha047.csv --test_path=data/imdb/test.csv \
         --log_dir=$HOME/crowd-embeddings/logs --checkpoint_root=$HOME/crowd-embeddings/checkpoints   \
         --backbone=lstm --approach=crowd_embedding   \
         --num_workers=8 \
         --max_epochs=25 --batch_size=32 \
         --inference_policy=top_k \
         --reproduction_iters=10 \
         --lr=0.0009886 --dropout=0.1 \
         --tune_iters=0 \
         --gpus=1 \
         2>&1 | tee train.log
```
