import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, \
    match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug

import os
import logging
import random
import torch.nn as nn


def build_logger(work_dir, cfgname):
    assert cfgname is not None
    log_file = cfgname + '.log'
    log_path = os.path.join(work_dir, log_file)

    logger = logging.getLogger(cfgname)
    logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler1 = logging.FileHandler(log_path)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)

    handler2 = logging.StreamHandler()
    handler2.setFormatter(formatter)
    logger.addHandler(handler2)
    logger.propagate = False

    return logger


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Decay the learning rate based on schedule"""
    lr = init_lr
    for milestone in [1200, 1600, 1800]: # 学习率衰减的里程碑点
        lr *= 0.5 if epoch >= milestone else 1. # 当达到里程碑时将学习率减半
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr # 更新优化器中所有参数组的学习率

def criterion_middle(real_feature, syn_feature):
    MSE_Loss = nn.MSELoss(reduction='sum')
    shape_real = real_feature.shape
    real_feature = torch.mean(real_feature.view(10, shape_real[0] // 10, *shape_real[1:]), dim=1)

    shape_syn = syn_feature.shape
    syn_feature = torch.mean(syn_feature.view(10, shape_syn[0] // 10, *shape_syn[1:]), dim=1)

    return MSE_Loss(real_feature, syn_feature)


def main():
    #允许用户在运行程序时通过命令行指定各种参数，比如使用哪个数据集、使用哪种模型、每类图像数量等，而不需要修改源代码。
    parser = argparse.ArgumentParser(description='Parameter Processing') #创建一个命令行参数解析器对象
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')  # 蒸馏方法选择
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')  # 使用的数据集
    parser.add_argument('--model', type=str, default='ConvNet', help='model')  # 训练使用的网络架构
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')  # 每类合成图像数量
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode')  # 评估模式: S-同训练模型, M-多架构, W-网络宽度, D-网络深度, A-激活函数, P-池化层, N-归一化层
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')  # 实验重复次数，用于结果稳定性
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')  # 每次评估使用的随机初始化模型数量
    parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data')  # 用合成数据训练模型的epoch数
    parser.add_argument('--Iteration', type=int, default=2000, help='training iterations')  # 总训练迭代次数，控制蒸馏过程的总轮数
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')  # 合成图像的学习率，影响图像更新速度
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')  # 网络参数的学习率，控制模型更新幅度
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')  # 真实数据的批大小，影响内存使用和梯度稳定性
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')  # 训练网络的批大小，与GPU显存容量相关
    parser.add_argument('--init', type=str, default='noise',
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')  # 初始化方式：noise-随机噪声 / real-真实图像采样
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')  # 可微分数据增强策略，具体策略内容需参考DSA论文
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')  # 原始数据集存储路径，需确保该路径包含对应数据集
    parser.add_argument('--save_path', type=str, default='oi_cifar10_ipc10_watcher_5_v3', help='path to save results')  # 结果保存路径，包含合成图像和评估结果
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric') # 特征距离度量方法
    parser.add_argument('--fourth_weight', type=float, default=0.1, help='batch size for training networks') # 网络第四层特征匹配的损失权重
    parser.add_argument('--third_weight', type=float, default=0.1, help='batch size for training networks') # 网络第三层特征匹配的损失权重
    parser.add_argument('--second_weight', type=float, default=1.0, help='batch size for training networks') # 网络第二层特征匹配的损失权重
    parser.add_argument('--first_weight', type=float, default=1.0, help='batch size for training networks') # 网络第一层特征匹配的损失权重 
    parser.add_argument('--inner_weight', type=float, default=0.01, help='batch size for training networks') # 内部循环损失项的加权系数

    parser.add_argument('--lambda_1', type=float, default=0.04, help='break outlooper') # 外层循环准确率波动阈值（代码329行）
    parser.add_argument('--lambda_2', type=float, default=0.03, help='break innerlooper') # 内层循环准确率波动阈值（代码365行）
    parser.add_argument('--gpu_id', type=str, default='0', help='dataset path') # 默认使用第一个可用的GPU
    
    #这行代码的作用是解析用户通过命令行输入的所有参数，并将结果存储在 args 对象中。
    args = parser.parse_args()
    logger = build_logger('.', cfgname=str(args.lambda_1) + "_" + str(args.lambda_2) + "_" + str(
        args.inner_weight) + '_' + str(args.fourth_weight) + '_' + str(args.third_weight) + '_' + str(
        args.second_weight) + '_' + str(args.first_weight) + 'oi_cifar10_dsa_ipc50_watcher_5_v3')  # 日志文件名格式：λ1_λ2_内部权重_第四层权重_第三层权重_第二层权重_第一层权重_实验标识
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id # 设置课件GPU
    args.outer_loop, args.inner_loop = get_loops(args.ipc) # 通过get_loops函数获取内外循环次数（参数基于每类图像数量ipc）
    # import pdb; pdb.set_trace()
    args.save_path = str(args.lambda_1) + "_" + str(args.lambda_2) + "_" + 'oi_cifar10_ipc10_watcher_5_v3' # 结果保存路径，包含合成图像和评估结果
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug() # 可微分数据增强参数
    args.dsa = True if args.method == 'DSA' else False # 是否使用可微分数据增强

    if not os.path.exists(args.data_path): # 创建数据集路径
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path): # 创建保存路径
        os.mkdir(args.save_path)

    # 根据评估模式创建评估点列表（S模式每100次迭代评估，其他模式只在最终迭代评估）
    eval_it_pool = np.arange(0, args.Iteration + 1, 100).tolist() if args.eval_mode == 'S' else [
        args.Iteration]  # 评估时机列表（代码224行会根据这个列表进行模型评估）
    
    # 获取数据集元数据和加载器（返回参数包含通道数、图像尺寸、类别数等关键信息）
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset,
                                                                                                         args.data_path)
    
    # 生成评估模型池（根据eval_mode参数决定包含哪些模型架构）
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)  # 模型评估池生成（代码228行使用）
    # 初始化准确率记录字典（结构：{模型名称: [准确率1, 准确率2,...]}）
    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []  # 为每个评估模型创建空列表，用于存储多次实验的测试准确率
    
    # 初始化合成数据保存列表（元素格式：[图像张量, 标签张量]）
    data_save = []  # 存储每次实验生成的最终合成数据
    
    
    
    
    for exp in range(args.num_exp): # 多次实验
        logger.info('================== Exp %d ==================' % exp) # 实验区块分隔线（带当前实验序号）
        # 记录完整的超参数配置（使用对象的__dict__属性转换为字典格式）
        # 注意：logger.info() 应接收单个字符串参数，建议使用格式字符串优化
        logger.info('Hyper-parameters: \n', args.__dict__) # 使用pprint美化输出
        print('Evaluation model pool: ', model_eval_pool) # 在控制台直接打印待评估模型池（与日志记录解耦，用于实时监控）

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)] # 使用二维列表实现 类别到样本索引的映射表，便于后续按类别快速采样

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))] # 在第0维增加batch维度
        labels_all = [dst_train[i][1] for i in range(len(dst_train))] # 提取所有样本的标签
        for i, lab in enumerate(labels_all): # 构建类别索引映射
            indices_class[lab].append(i) # 记录每个类别对应的样本索引
        # 转换为张量
        images_all = torch.cat(images_all, dim=0) # 合并所有图像张量 (num_samples, C, H, W)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device) # 标签转换为设备上的长整型张量
        # 记录类别分布信息
        for c in range(num_classes):
            logger.info('class c = %d: %d real images' % (c, len(indices_class[c]))) # 记录类别对应索引号，即分布信息

        def get_images(c, n):  # get random n images from class c
            # import pdb; pdb.set_trace()
            idx_shuffle = np.random.permutation(indices_class[c])[:n] # 随机打乱后取前n个
            return images_all[idx_shuffle].to(args.device) # 移动数据到指定设备（如GPU）

        for ch in range(channel): # 分析图像通道统计量
            logger.info('real images channel %d, mean = %.4f, std = %.4f' % (
                ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch]))) # 格式化输出保留4位小数，建议添加通道含义说明（如RGB对应通道）

        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float,
                                requires_grad=True, device=args.device) # 创建可学习的合成图像数据，使用标准正态分布（均值为 0，标准差为 1）的随机噪声初始化
        label_syn = torch.tensor([np.ones(args.ipc) * i for i in range(num_classes)], dtype=torch.int,
                                 requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9] 生成与合成图像对应的类别标签

        if args.init == 'real': # 初始化合成数据为随机真实图像
            logger.info('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data # 从真实图像中随机选取args.ipc个图像，并将它们的数据赋值给合成图像
        else:
            logger.info('initialize synthetic data from random noise')

        ''' training '''
        # 配置优化器以更新合成数据 image_syn
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)  # optimizer_img for synthetic data
        optimizer_img.zero_grad() # 清空优化器的梯度缓存
        criterion = nn.CrossEntropyLoss().to(args.device) # 定义损失函数为交叉熵损失函数，并将其移动到指定的设备上
        criterion_sum = nn.CrossEntropyLoss(reduction='sum').to(args.device) 
        logger.info('%s training begins' % get_time()) # 记录训练开始时间




# 开始每一次实验中的蒸馏过程，循环，每一次都是单次迭代
        for it in range(args.Iteration + 1): 
            adjust_learning_rate(optimizer_img, it, args.lr_img) # 调整学习率
    
            ''' Evaluate synthetic data '''
            if it in eval_it_pool:  # 只在预定义的评估点执行评估
                for model_eval in model_eval_pool:  # 遍历所有评估模型架构
                    # 打印评估头信息（当前训练模型/评估模型/迭代次数）
                    logger.info(
                        '-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
                            args.model, model_eval, it))
                    if args.dsa:  # 如果使用可微分数据增强  # 使用DC方法的数据增强策略，具体策略内容需参考DSA论文  # 获取当前模型的数据增强参数（仅限DC方法）
                        args.epoch_eval_train = 1000  # 设置较长的训练周期
                        args.dc_aug_param = None  # 禁用DC方法的数据增强
                        # 记录DSA增强策略和参数
                        logger.info('DSA augmentation strategy: \n' + args.dsa_strategy)
                        logger.info('DSA augmentation parameters: \n' + str(args.dsa_param.__dict__))
                    else:  # 使用DC方法的数据增强
                        # 获取当前模型的数据增强参数（仅限DC方法）
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval,
                                                        args.ipc)  
                        logger.info('DC augmentation parameters: \n' + str(args.dc_aug_param))

                    # 根据是否使用数据增强调整训练周期
                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # 使用增强时需要更长的训练周期
                    else:
                        args.epoch_eval_train = 600  # 无增强时使用较短周期

                    # 评估模型的性能
                    accs = []
                    for it_eval in range(args.num_eval):
                        # get a random model
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(
                            args.device)  
                        # avoid any unaware modification
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                            label_syn.detach())  
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval,testloader, args)
                        accs.append(acc_test)
                    logger.info('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                        len(accs), model_eval, np.mean(accs), np.std(accs)))
                    # record the final results
                    if it == args.Iteration:  
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                # 构建可视化文件保存路径（包含方法/数据集/模型/IPC/实验号/迭代次数等信息）
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png' % (
                    args.method, args.dataset, args.model, args.ipc, exp, it))
                # 创建合成数据副本并转移到CPU（避免影响GPU上的训练过程）
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                # 反归一化处理：恢复原始像素值范围（基于数据集的均值标准差）
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                # 像素值截断到[0,1]区间（防止浮点运算误差导致越界）
                image_syn_vis[image_syn_vis < 0] = 0.0
                image_syn_vis[image_syn_vis > 1] = 1.0
                # 保存合成图像网格（nrow控制每行显示图像数量，对应每类样本数）
                save_image(image_syn_vis, save_name,nrow=args.ipc)  

            ''' Train synthetic data '''
            # get a random model
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)  
            net.train()  # 将模型设置为训练模式
            net_parameters = list(net.parameters())  # 获取模型参数列表
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # 为模型参数创建SGD优化器
            optimizer_net.zero_grad()  # 清空优化器的梯度缓存
            loss_avg = 0  # 初始化平均损失
            loss_kai = 0  # 初始化特定损失项
            loss_middle_item = 0  # 初始化中间损失项
            args.dc_aug_param = None  # 训练合成数据时禁用DC数据增强
            
            # for ol in range(args.outer_loop):  # 外层循环（已注释，改用while True）
            acc_watcher = list()  # 初始化准确率观察列表，用于监控模型性能
            pop_cnt = 0  # 初始化弹出计数器
            acc_test = 0.0  # 初始化测试准确率

            while True:  # 无限循环，直到满足退出条件
                syn_centers = []  # 初始化合成数据特征中心列表
                real_feature_concat = []  # 初始化真实数据特征列表
                real_feature_concat_mm = []  # 初始化真实数据特征（多模态）列表
                real_label_concat = []  # 初始化真实数据标签列表
                img_real_gather = []  # 初始化真实图像收集列表
                img_syn_gather = []  # 初始化合成图像收集列表
                lab_real_gather = []  # 初始化真实标签收集列表
                lab_syn_gather = []  # 初始化合成标签收集列表
                
                loss = torch.tensor(0.0).to(args.device)  # 初始化损失张量，并移动到指定设备
                for c in range(num_classes):  # 遍历所有类别
                    img_real = get_images(c, args.batch_real)  # 获取当前类别的真实图像样本
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c  # 创建对应的真实标签
                    img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape(
                        (args.ipc, channel, im_size[0], im_size[1]))  # 获取当前类别的合成图像
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c  # 创建对应的合成标签
                    
                    
                    if args.dsa:  # 如果启用了可微分数据增强
                        seed = int(time.time() * 1000) % 100000  # 生成随机种子
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)  # 对真实图像进行数据增强
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)  # 对合成图像进行数据增强
                    img_real_gather.append(img_real)  # 将处理后的真实图像添加到收集列表
                    lab_real_gather.append(lab_real)  # 将真实标签添加到收集列表
                    img_syn_gather.append(img_syn)  # 将处理后的合成图像添加到收集列表
                    lab_syn_gather.append(lab_syn)  # 将合成标签添加到收集列表

                img_real_gather = torch.stack(img_real_gather, dim=0).reshape(args.batch_real * 10, 3, 32, 32)  # 将真实图像堆叠并重塑为指定形状
                img_syn_gather = torch.stack(img_syn_gather, dim=0).reshape(args.ipc * 10, 3, 32, 32)  # 将合成图像堆叠并重塑为指定形状
                lab_real_gather = torch.stack(lab_real_gather, dim=0).reshape(args.batch_real * 10)  # 将真实标签堆叠并重塑为指定形状
                lab_syn_gather = torch.stack(lab_syn_gather, dim=0).reshape(args.ipc * 10)  # 将合成标签堆叠并重塑为指定形状

                ####forward#####
                # 前向传播：将真实图像输入网络，获取输出和特征
                output_real, real_features = net(img_real_gather)
                # 前向传播：将合成图像输入网络，获取输出和特征
                output_syn, syn_features = net(img_syn_gather)
                # 计算中间损失
                loss_middle = args.fourth_weight * criterion_middle(real_features[-1], syn_features[-1]) + args.third_weight * criterion_middle(real_features[-2], syn_features[-2]) + args.second_weight * criterion_middle(real_features[-3], syn_features[-3]) + args.first_weight * criterion_middle(real_features[-4], syn_features[-4])
                loss_real = criterion(output_real, lab_real_gather)
                # 累加中间损失到总损失
                loss += loss_middle
                # 累加真实损失到总损失
                loss += loss_real

                # 计算第一层特征的平均值（按类别分组后取平均）
                last_real_feature = torch.mean(real_features[0].view(10, int(real_features[0].shape[0] / num_classes), real_features[0].shape[1]), dim=1)
                # 计算合成图像第一层特征的平均值
                last_syn_feature = torch.mean(syn_features[0].view(10, int(syn_features[0].shape[0] / num_classes), syn_features[0].shape[1]), dim=1)
                # 计算真实特征和合成特征之间的矩阵乘法
                output = torch.mm(real_features[0], last_syn_feature.t())
                # 对真实特征进行进一步reshape和平均操作
                last_real_feature = torch.mean(
                    last_real_feature.unsqueeze(0).reshape(10, int(last_real_feature.shape[0] / num_classes),last_real_feature.shape[1]), dim=1)
                # 计算输出损失：特征匹配损失 + 加权分类损失
                loss_output = criterion_middle(last_syn_feature, last_real_feature) + args.inner_weight * criterion_sum(output, lab_real_gather)
                # 累加输出损失到总损失
                loss += loss_output

                # 反向传播计算梯度
                loss.backward()
                # 更新合成图像参数
                optimizer_img.step()
                # 清空梯度缓存
                optimizer_img.zero_grad()
                # 累加总损失用于后续统计
                loss_avg += loss.item()
                # 累加输出损失用于后续统计
                loss_kai += loss_output.item()
                # 累加中间损失用于后续统计
                loss_middle_item += loss_middle.item()
                
                ############ for outloop testing ############

                # 遍历所有类别
                for c in range(num_classes):
                    # 获取当前类别的128个真实图像样本
                    img_real_test = get_images(c, 128)
                    # 创建对应的真实标签
                    lab_real_test = torch.ones((img_real_test.shape[0],), device=args.device, dtype=torch.long) * c
                    # 将图像输入网络，获取预测概率
                    prob, _ = net(img_real_test)
                    # 计算当前类别的准确率并累加
                    acc_test += (lab_real_test == prob.max(dim=1)[1]).float().mean()
                # 计算所有类别的平均准确率
                acc_test /= num_classes
                # 将当前准确率添加到观察列表
                acc_watcher.append(acc_test.detach().cpu())
                # 增加弹出计数器
                pop_cnt += 1
                # 当观察列表达到10个值时
                if len(acc_watcher) == 10:
                    # 如果最大和最小准确率差值小于阈值lambda_1
                    if max(acc_watcher) - min(acc_watcher) < args.lambda_1:
                        # 重置观察列表
                        acc_watcher = list()
                        # 重置计数器
                        pop_cnt = 0
                        # 重置准确率
                        acc_test = 0.0
                        # 退出外层循环
                        break
                    else:
                        # 否则移除最旧的准确率记录
                        acc_watcher.pop(0)


                ''' update network '''
                # 创建合成数据和标签的深拷贝，避免后续操作影响原始数据
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
                # 将合成数据包装成TensorDataset
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                # 创建DataLoader，用于后续训练
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                # 初始化内层循环的准确率观察列表
                acc_inner_watcher = list()
                # 初始化合成数据训练的准确率观察列表
                acc_syn_inner_watcher = list()
                # 初始化内层循环的弹出计数器
                pop_inner_cnt = 0
                # 初始化内层循环的测试准确率
                acc_inner_test = 0
                
                # for il in range(args.inner_loop):  # 原注释的内层循环，已被while循环替代
                while (1):  # 无限循环，直到满足退出条件
                    # 使用epoch函数进行训练，返回损失和准确率
                    inner_loss, inner_acc = epoch('train', trainloader, net, optimizer_net, criterion, args, aug=True if args.dsa else False)
                    # 将训练准确率添加到观察列表
                    acc_syn_inner_watcher.append(inner_acc)
                    
                    # 遍历所有类别进行测试
                    for c in range(num_classes):
                        # 获取当前类别的128个真实图像样本
                        img_real_test = get_images(c, 128)
                        # 创建对应的真实标签
                        lab_real_test = torch.ones((img_real_test.shape[0],), device=args.device, dtype=torch.long) * c
                        # 将图像输入网络，获取预测概率
                        prob, _ = net(img_real_test)
                        # 计算当前类别的准确率并累加
                        acc_inner_test += (lab_real_test == prob.max(dim=1)[1]).float().mean()
                    
                    # 计算所有类别的平均准确率
                    acc_inner_test /= num_classes
                    # 将当前准确率添加到观察列表
                    acc_inner_watcher.append(acc_inner_test.detach().cpu())
                    # 增加弹出计数器
                    pop_inner_cnt += 1
                    
                    # 当观察列表达到10个值时
                    if len(acc_inner_watcher) == 10:
                        # 如果最大和最小准确率差值大于阈值lambda_2
                        if max(acc_inner_watcher) - min(acc_inner_watcher) > args.lambda_2:
                            # 重置所有监控变量
                            acc_inner_watcher = list()
                            acc_syn_inner_watcher = list()
                            pop_inner_cnt = 0
                            acc_inner_test = 0
                            # 退出内层循环
                            break
                        else:
                            # 否则移除最旧的准确率记录
                            acc_inner_watcher.pop(0)

                    epoch('test', trainloader, net, optimizer_net, criterion, args, aug=True if args.dsa else False) # 测试模型性能

            loss_avg /= (num_classes * args.outer_loop) # 归一化为单类别单次外层循环的平均损失

            if it % 10 == 0: # 每10次迭代记录一次
                logger.info('%s iter = %04d, loss = %.4f, loss_kai = %.4f, loss_middle = %.4f' % (
                    get_time(), it, loss_avg, loss_kai, loss_middle_item)) # 记录当前时间、迭代次数、平均损失、特定损失和中间损失

            if it == args.Iteration:  # 仅在达到最大迭代次数时执行
                # 将当前合成数据和标签的深拷贝（detach并转移到CPU）添加到data_save列表
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                # 保存结果到文件，包含：
                # 1. 合成数据 (data_save)
                # 2. 所有评估模型的准确率 (accs_all_exps)
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, },os.path.join(args.save_path,'res_%s_%s_%s_%dipc.pt' % (args.method,args.dataset,args.model,args.ipc)))
    
    logger.info('\n==================== Final Results ====================\n') # 打印结果
    for key in model_eval_pool: # 遍历所有评估模型
        accs = accs_all_exps[key]
        # 输出统计信息：实验次数、训练模型、评估模型、平均准确率和标准差
        logger.info('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%' % (
            args.num_exp, args.model, len(accs), key, np.mean(accs) * 100, np.std(accs) * 100))


if __name__ == '__main__':
    main()
