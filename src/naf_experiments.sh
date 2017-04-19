#!/bin/bash

python pendulum_naf.py --episodes 300 --tau 0.001 --noise epsilon_greedy --expert_model ./results/pendulum/01_Jan_2017_18_18_25 --noise_scale 0.2 --optimizer_lr 0.0005 --d_lr 0.00001 --d_sampling False --notes "no discriminator replay memory"

#python pendulum_naf.py --episodes 150 --priority_prob 0.5 --optimizer_lr 0.0002 --d_lr 0.000006 --notes "low D"
#python pendulum_naf.py --episodes 150 --priority_prob 0.5 --optimizer_lr 0.0002 --d_lr 0.00003 --notes "high D"
#python pendulum_naf.py --episodes 150 --priority_prob 0.5 --optimizer_lr 0.0005 --d_lr 0.000006 --notes "high Q"
#python pendulum_naf.py --episodes 150 --priority_prob 0.5 --optimizer_lr 0.0001 --d_lr 0.000006 --notes "low Q"
#python pendulum_naf.py --episodes 150 --priority_prob 0.5 --optimizer_lr 0.0005 --d_lr 0.00005 --notes "high Q and D"
#python pendulum_naf.py --episodes 150 --priority_prob 0.5 --optimizer_lr 0.0005 --d_lr 0.00001 --notes "high Q and D"
#python pendulum_naf.py --episodes 150 --priority_prob 0.5 --optimizer_lr 0.0007 --d_lr 0.00003 --notes "high Q and D"
#python pendulum_naf.py --episodes 150 --priority_prob 0.5 --optimizer_lr 0.0003 --d_lr 0.00003 --notes "high Q and D"

#python pendulum_naf.py --episodes 150 --priority_prob 0.5 --optimizer_lr 0.0001 --d_lr 0.000006 --notes "low Q and D"
#python pendulum_naf.py --episodes 150 --priority_prob 1. --notes "one priority"
#python pendulum_naf.py --episodes 150 --priority_prob 0. --notes "zero priority"
#python pendulum_naf.py --episodes 150 --tau 0.005 --notes "larger tau"
#python pendulum_naf.py --episodes 200 --tau 0.0003 --notes "smaller tau"


#python pendulum_naf.py --episodes 150 --tau 0.001 --noise exp_decay --noise_scale 0.008 --optimizer_lr 0.0005 --d_lr 0.00001 --notes "fast exponential exploration decay"

#python pendulum_naf.py --episodes 150 --tau 0.001 --optimizer_lr 0.0005 --d_lr 0.00001 --notes "higher Q learning rate"

#python pendulum_naf.py --episodes 150 --tau 0.001 --optimizer_lr 0.0002 --d_lr 0.00005 --notes "higher D learning rate"

#python pendulum_naf.py --episodes 200 --tau 0.001 --optimizer_lr 0.0005 --d_lr 0.00005 --notes "higher D and Q learning rate"

#python pendulum_naf.py --episodes 150 --tau 0.001 --optimizer_lr 0.00005 --d_lr 0.00001 --notes "Lower Q learning rate"

#python pendulum_naf.py --episodes 150 --tau 0.001 --noise_scale 0.05 --optimizer_lr 0.0005 --d_lr 0.00001 --notes "faster exploration decay"


#python pendulum_naf.py --episodes 150 --tau 0.001 --noise exp_decay --noise_scale 0.005 --optimizer_lr 0.0005 --d_lr 0.00001 --notes "slow exponential exploration decay"

#python pendulum_naf.py --episodes 150 --tau 0.001 --noise epsilon_greedy --noise_scale 0.2 --optimizer_lr 0.0005 --d_lr 0.00001 --notes "epsilon greedy with decay"


#python pendulum_naf.py --episodes 150 --batch_size 100 --tau 0.001 --optimizer_lr 0.0005 --d_lr 0.00001 --notes "smaller batch size"

#python pendulum_naf.py --episodes 150 --l2_reg 0.0001 --tau 0.001 --optimizer_lr 0.0005 --d_lr 0.00001 --notes "l2 regularisation"
#python pendulum_naf.py --episodes 150 --l1_reg 0.0001 --tau 0.001 --optimizer_lr 0.0005 --d_lr 0.00001 --notes "l1 regularisation"


#python pendulum_naf.py --episodes 150 --tau 0.001 --optimizer_lr 0.0002 --d_lr 0.000005 --notes "lower D learning rate"
#python pendulum_naf.py --environment MountainCarContinuous-v0 --expert_model ./results/pendulum/04_Jan_2017_21_36_43  --episodes 200 --tau 0.001 --optimizer_lr 0.0001 --max_timesteps 2000 --d_lr 0.00001 --noise epsilon_greedy --noise_scale 0.2 --notes "mountain car learning with expert"

#python pendulum_naf.py --environment Acrobot-v1  --episodes 200 --tau 0.001 --priority_prob 0.5 --optimizer_lr 0.0001 --max_timesteps 20 --d_lr 0.00001 --noise epsilon_greedy --noise_scale 0.2 --notes "Acrobot v1"

