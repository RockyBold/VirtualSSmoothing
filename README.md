# Training Deep Neural Networks with Virtual Smoothing Classes

**Requisite**\
This code is implemented in PyTorch, and we have tested the code under the following environment settings:\
`python = 3.8.5`\
`numpy == 1.19.2`\
`torch = 1.7.1`\
`torchvision = 0.8.2`

**Train Neural Networks with Virtual Smoothing Labels**
1. Standard Training:  
`python clean_train_vs.py --model_name resnet-18 --dataset cifar10 --training_method clean --v_classes 10 --alpha 0.5 --gpuid 0 --model_dir $MODEL_DIR$`

2. Adversarial Training:  
`python rob_train_vs.py --model_name WRN-34-10 --dataset cifar10 --training_method pgd --v_classes 10 --alpha 0.9 --gpuid 0 --model_dir $MODEL_DIR$`

**Evaluation**    
`python eval_rob --gpuid 0 --v_nodes 10 --model_file $MODEL_DIR$`

**Reference Code**\    
[1] AT: https://github.com/locuslab/robust_overfitting \
[2] TRADES: https://github.com/yaodongyu/TRADES/ \
[3] AutoAttack: https://github.com/fra31/auto-attack \
[4] RST: https://github.com/yaircarmon/semisup-adv \
[5] AWP: https://github.com/csdongxian/AWP




