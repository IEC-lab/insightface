python -u verification.py \
           --gpu 0 \
           --data-dir /home/data/insight/insight/faces_ms1m_112x112 \
           --model '../../models/am_r34_1024/model,81|91' \
           --target lfw \
           --batch-size 64
