#### 实验流程：

1. 准备数据集：训练集分为 input 和 target 两个路径，测试集同理
2. 修改参数配置文件：打开 configs.yml 文件，修改其中的数据集路径
3. 改变预训练模型存储位置：见 configs.yml 中的 training.resume 值
4. 确认自己的电脑可以使用 cuda，如果只有 cpu 的话跑不起来
5. 运行 train_diffusion.py 文件，判断在当前参数配置下是否可以跑起来
6. 根据实际情况修改 configs.yml 中的 training.batch_size 值
7. 根据实际需求修改训练所需要的次数，默认 epoch 为 2000，见 configs.yml 中的 training.n_epochs 值
8. 训练过程中会将验证集上的结果保存在 validation 文件夹中
9. 训练结束
9. 在测试前建议先将所有测试集中的图片大小尺寸变为 16 的倍数
10. 这里要注意测试路径里也要同时有 input 和 target 两个路径，虽然 target 不参与测试过程
10. 运行 test_diffusion.py 文件，耐心等待，每一幅图片大概需要几十秒
11. 测试结束
12. 直接运行 calculate_psnr_ssim.py 计算指标