PYTHONPATH=. python detector/fasterrcnn.py --epochs 100 --lr 0.0001 --batch-size 16 --img-size 1024 --device cuda:1

PYTHONPATH=. python detector/fasterrcnn.py --epochs 100 --lr 0.0001 --batch-size 16 --img-size 800 --device cuda:1

PYTHONPATH=. python detector/fasterrcnn.py --epochs 100 --lr 0.0001 --batch-size 16 --img-size 512 --device cuda:1

PYTHONPATH=. python detector/fasterrcnn.py --epochs 100 --lr 0.0001 --batch-size 16 --img-size 224 --device cuda:1

