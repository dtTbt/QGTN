### 0. Access Screen and Resume Session
```bash
screen -r QGTN
```

### 1. Multi-GPU Training from Compressed Archive
```bash
rm -rf ../QGTN
rm -rf ../QGTN.zip
mv ./QGTN.zip ../
cd ../
unzip QGTN.zip
cd QGTN
screen -wipe
screen -X -S QGTN quit
screen -U -S QGTN
conda activate coat
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use_env \
    main.py \
    --batch_size 10
```

### 2. Direct Multi-GPU Training
```bash
screen -wipe
screen -X -S QGTN quit
screen -U -S QGTN
cd ../QGTN
conda activate coat
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env \
    main.py \
    --batch_size 10
```

### 3. Single-GPU Training
```bash
screen -wipe
screen -X -S QGTN quit
screen -U -S QGTN
cd ../QGTN
conda activate coat
python main.py
```

### 4. Single-GPU Testing
```bash
screen -wipe
screen -X -S QGTN quit
screen -U -S QGTN
cd ../QGTN
conda activate coat
python main.py --batch_size 8 --eval
```
