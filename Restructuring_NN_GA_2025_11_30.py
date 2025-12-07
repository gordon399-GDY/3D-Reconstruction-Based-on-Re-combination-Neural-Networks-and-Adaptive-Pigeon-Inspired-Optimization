# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 22:15:54 2023
@author: Dongyuan Ge
优化版本：基于成功经验的遗传算法优化输入矢量
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time

start_time = time.time()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)  # 设置固定种子以获得稳定结果


# 生成区间[a,b]内的随机数
def random_number(a, b):
    return (b - a) * random.random() + a


# 生成一个矩阵，大小为m*n,并且设置默认零矩阵
def makematrix(m, n, fill=0.0):
    a = []
    for i in range(m):
        a.append([fill] * n)
    return a


# 函数sigmoid(),这里采用tanh
def sigmoid(x):
    return 1 * (math.tanh(x))


# 函数sigmoid的派生函数
def derived_sigmoid(x):
    return 1 * (1.0 - x ** 2)


# 构造三层BP网络架构
class BPNN:
    def __init__(self, num_in, num_hidden, num_out):
        # 输入层，隐藏层，输出层的节点数
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_out = num_out

        # 激活神经网络的所有节点（向量）
        self.active_in = [1.0] * self.num_in
        self.active_hidden = [1.0] * self.num_hidden
        self.active_out = [1.0] * self.num_out

        # 创建权重矩阵
        self.wight_in = makematrix(self.num_in, self.num_hidden)
        self.wight_out = makematrix(self.num_hidden, self.num_out)

        # 对权值矩阵赋初值
        for i in range(self.num_in):
            for j in range(self.num_hidden):
                self.wight_in[i][j] = random_number(-0.9, 0.9)
        for j in range(self.num_hidden):
            for k in range(self.num_out):
                self.wight_out[j][k] = random_number(-0.9, 0.9)

        self.wight_in = [
            [-2.51959160426336, -2.751364742571889, 7.654851540359079, -4.211016719195278, 2.4456066086640993,
             -5.102025306470403, 1.3187086463134194, -1.5078691162192488, 1.7462421090646636, 0.7574105748038743,
             0.4749123220695475, -0.25942496346826793],
            [-2.074356719185996, -2.103076309704452, 1.0235356936830249, -5.234417980661577, -0.0652419463854865,
             -9.629819382253665, -1.124932941978192, -3.787769703518782, 0.5698050055464846, -0.12346253664064348,
             -3.441546495129915, -0.2710410400988457],
            [-8.23974589637843, 5.5913214930121695, -37.999575190461805, 2.688322198642766, 9.674410269139834,
             3.1314409317445286, 0.2279391532919728, 5.365653638177657, -25.83385572991718, 15.448858184956697,
             9.936119709810107, 6.557420466859034],
            [0.744876427326448, 1.0959850758758258, -3.076768819457162, -1.4619364380827036, -1.8780111033404125,
             0.3393397595076832, 0.33066242156159953, 0.5599058867821437, -1.892048952693372, -0.48233646869150837,
             -0.9482582130242485, -0.2562522533269894],
        ]
        self.wight_out = [

            [-0.8311159523930649, -7.200999601851575, 0, 0],
            [-3.093623801650995, 1.9546485390558268, 0, 0],
            [0.654137851147521, -0.12859264818489702, 0, 0],
            [1.9804415012925316, -3.2170426682683475, 0, 0],
            [1.678694433992265, -7.3505336133828365, 0, 0],
            [-0.33424864811331223, 1.848669231247442, 0, 0],

            [0, 0, 1.7945104598375976, 0.7497905597477524],
            [0, 0, -0.6768732350426181, -0.034311744099842216],
            [0, 0, 1.4462304768082141, 0.13285312733990573],
            [0, 0, 2.568594657996465, 0.7287958056936737],
            [0, 0, 0.2875715209256103, -2.48305847472072],
            [0, 0, -1.766237611544275, 2.5450906267101807],
        ]
        # 动量因子（矩阵）
        self.ci = makematrix(self.num_in, self.num_hidden)
        self.co = makematrix(self.num_hidden, self.num_out)

        # 存储每个矢量的误差数据
        self.vector_errors = []

    def update(self, inputs):
        if len(inputs) != self.num_in:
            raise ValueError('与输入层节点数不符')

        # 数据输入输入层
        for i in range(self.num_in):
            self.active_in[i] = inputs[i]

        # 数据在隐藏层的处理
        for j in range(self.num_hidden):
            sum_val = 0.0
            for i in range(self.num_in):
                sum_val = sum_val + self.active_in[i] * self.wight_in[i][j]
            self.active_hidden[j] = sigmoid(sum_val)

        # 数据在输出层的处理
        for k in range(self.num_out):
            sum_val = 0.0
            for j in range(self.num_hidden):
                sum_val = sum_val + self.active_hidden[j] * self.wight_out[j][k]
            self.active_out[k] = sigmoid(sum_val)

        return self.active_out[:]

    def calculate_error(self, inputs, targets):
        """计算给定输入和目标输出的均方误差"""
        outputs = self.update(inputs)
        error = 0.0
        for i in range(len(targets)):
            error += 0.5 * (targets[i] - outputs[i]) ** 2
        return error

    def errorbackpropagate(self, targets, lr, m):
        if len(targets) != self.num_out:
            raise ValueError('与输出层节点数不符！')

        # 计算输出层的误差
        out_deltas = [0.0] * self.num_out
        for k in range(self.num_out):
            error = targets[k] - self.active_out[k]
            out_deltas[k] = derived_sigmoid(self.active_out[k]) * error

        # 计算隐藏层误差
        hidden_deltas = [0.0] * self.num_hidden
        for j in range(self.num_hidden):
            error = 0.0
            for k in range(self.num_out):
                error = error + out_deltas[k] * self.wight_out[j][k]
            hidden_deltas[j] = derived_sigmoid(self.active_hidden[j]) * error

        # 更新输入信息
        for i in range(self.num_in - 1):
            change = [0, 0, 0]
            for j in range(self.num_hidden):
                change[i] = change[i] + hidden_deltas[j] * self.wight_in[i][j]
            self.active_in[i] = self.active_in[i] + lr * change[i]

        # 计算总误差
        error = 0.0
        for i in range(len(targets)):
            error = error + 0.5 * (targets[i] - self.active_out[i]) ** 2
        return error

    def test(self, patterns):
        for i in patterns:
            print(i[0], '->', self.update(i[0]))

    def weights(self):
        print("输入层权重")
        for i in range(self.num_in):
            print(self.wight_in[i])
        print("输出层权重")
        for i in range(self.num_hidden):
            print(self.wight_out[i])

    def plot_errors(self):
        if not self.vector_errors:
            print("没有误差数据可绘制")
            return

        indices = [data[0] for data in self.vector_errors]
        errors = [data[1] for data in self.vector_errors]

        plt.figure(figsize=(12, 6))
        plt.plot(indices, errors, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('采样点 (Sampling points)', fontsize=12)
        plt.ylabel('性能指标  (Performance indices)', fontsize=12)
        plt.title('系统训练的性能指标 (Performance indices of the system training)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, len(indices) + 1, 10))
        plt.tight_layout()
        plt.show()

    def train(self, pattern, itera=100001, lr=0.00930, m=0.00):
        self.vector_errors = []

        for k, j in enumerate(pattern):
            for i in range(itera):
                error = 0.0
                if i == 0:
                    inputs = j[0]
                    targets = j[1]
                    self.update(inputs)
                    error = error + self.errorbackpropagate(targets, lr, m)
                else:
                    inputs = [self.active_in[0], self.active_in[1], self.active_in[2], 0.0001]
                    targets = j[1]
                    self.update(inputs)
                    error = error + self.errorbackpropagate(targets, lr, m)

                if (i + 1) % 100000 == 0:
                    print(k + 1, error * 10000, "[", inputs[0] * 10000, inputs[1] * 10000, inputs[2] * 10000, "]")

                if (i + 1) == 100000: #this 100000 can be replace the available itera
                        self.vector_errors.append((k + 1, error * 10000))


class AdvancedGeneticAlgorithmOptimizer:
    def __init__(self, nn, population_size=500, max_generations=1500,
                 mutation_rate=0.05, crossover_rate=0.92, elite_size=20):
        self.nn = nn
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size

        # 收敛参数 - 调整为更严格的标准
        self.convergence_threshold = 1e-12  # 更严格的收敛阈值
        self.stagnation_generations = 100  # 更长的停滞判断周期

    def create_individual(self):
        """创建一个个体（前三个输入变量）"""
        return [random.uniform(0, 1) for _ in range(3)]

    def create_population(self):
        """创建初始种群"""
        return [self.create_individual() for _ in range(self.population_size)]

    def fitness(self, individual, targets):
        """计算适应度（误差越小适应度越高）"""
        # 构建完整输入：前三个优化变量 + 固定值0.0001
        inputs = individual + [0.0001]
        error = self.nn.calculate_error(inputs, targets)

        # 使用更敏感的适应度函数 - 针对小误差优化
        if error < 1e-20:
            return 1e30
        elif error < 1e-15:
            return 1e25
        elif error < 1e-12:
            return 1e20
        elif error < 1e-10:
            return 1e15
        elif error < 1e-8:
            return 1e10
        else:
            # 使用指数函数增强小误差的适应度差异
            return 1.0 / (error + 1e-20)

    def rank_population(self, population, targets):
        """对种群进行排序"""
        fitness_scores = [(ind, self.fitness(ind, targets)) for ind in population]
        return sorted(fitness_scores, key=lambda x: x[1], reverse=True)

    def selection(self, ranked_population):
        """改进的锦标赛选择"""
        # 80%锦标赛选择，20%轮盘赌选择
        if random.random() < 0.8:
            # 锦标赛选择 - 增加锦标赛规模
            selection_size = 8
            tournament = random.sample(ranked_population, selection_size)
            winner = max(tournament, key=lambda x: x[1])
            return winner[0]
        else:
            # 轮盘赌选择
            total_fitness = sum(fit for _, fit in ranked_population)
            if total_fitness == 0:
                return random.choice(ranked_population)[0]
            pick = random.uniform(0, total_fitness)
            current = 0
            for ind, fit in ranked_population:
                current += fit
                if current > pick:
                    return ind
            return ranked_population[0][0]

    def crossover(self, parent1, parent2):
        """改进的交叉策略"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        if random.random() < self.crossover_rate:
            # 多种交叉策略
            crossover_type = random.random()
            if crossover_type < 0.4:
                # 算术交叉 - 增加多样性
                alpha = random.uniform(-0.2, 1.2)  # 扩展范围以增加多样性
                for i in range(3):
                    child1[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
                    child2[i] = alpha * parent2[i] + (1 - alpha) * parent1[i]
            elif crossover_type < 0.7:
                # 单点交叉
                crossover_point = random.randint(1, 2)
                for i in range(crossover_point, 3):
                    child1[i], child2[i] = child2[i], child1[i]
            else:
                # 均匀交叉
                for i in range(3):
                    if random.random() < 0.5:
                        child1[i], child2[i] = child2[i], child1[i]

        return child1, child2

    def mutate(self, individual, generation, max_generations):
        """改进的自适应变异策略"""
        mutated = individual.copy()

        # 自适应变异率 - 在后期使用更小的变异
        progress = generation / max_generations
        adaptive_mutation_rate = self.mutation_rate * (1 - progress ** 2)  # 非线性衰减

        for i in range(3):
            if random.random() < adaptive_mutation_rate:
                # 多种变异策略
                mutation_type = random.random()
                if mutation_type < 0.4:
                    # 高斯变异 - 使用更小的标准差
                    mutation_strength = random.gauss(0, 0.01 * (1 - progress))
                    mutated[i] += mutation_strength
                elif mutation_type < 0.7:
                    # 均匀变异
                    mutated[i] = random.uniform(0, 1)
                else:
                    # 边界变异 - 更温和的边界变异
                    if random.random() < 0.5:
                        mutated[i] += 0.05 * (1 - mutated[i])
                    else:
                        mutated[i] -= 0.05 * mutated[i]

                # 确保值在[0,1]范围内
                mutated[i] = max(0, min(1, mutated[i]))
        return mutated

    def evolve(self, targets, enable_restart=False, max_restarts=3):
        """执行高级遗传算法进化过程"""
        best_individual_overall = None
        best_fitness_overall = -float('inf')
        best_error_overall = float('inf')
        best_fitness_history = []
        best_error_history = []

        total_restarts_used = 0
        convergence_achieved = False

        print("开始高级遗传算法优化...")

        for restart in range(max_restarts if enable_restart else 1):
            if convergence_achieved:
                break

            if enable_restart and restart > 0:
                print(f"--- 第{restart + 1}次重启 ---")
                total_restarts_used += 1

            population = self.create_population()
            best_individual = None
            best_fitness = -float('inf')
            best_error = float('inf')
            error_history = []
            stagnation_count = 0
            last_improvement = 0

            for generation in range(self.max_generations):
                # 评估种群
                ranked_population = self.rank_population(population, targets)

                # 更新最佳个体
                current_best_individual, current_best_fitness = ranked_population[0]
                current_best_inputs = current_best_individual + [0.0001]
                current_best_error = self.nn.calculate_error(current_best_inputs, targets)

                if current_best_fitness > best_fitness:
                    best_fitness = current_best_fitness
                    best_individual = current_best_individual.copy()
                    best_error = current_best_error
                    stagnation_count = 0
                    last_improvement = generation
                else:
                    stagnation_count += 1

                error_history.append(current_best_error)

                # 记录历史
                best_fitness_history.append(current_best_fitness)
                best_error_history.append(current_best_error)

                # 输出进度
                if (generation + 1) % 200 == 0:
                    progress = (generation + 1) / self.max_generations * 100
                    print(f"  第{generation + 1}代, 进度: {progress:.1f}%, 最佳误差: {best_error:.12f}")

                # 检查收敛条件
                if best_error < self.convergence_threshold:
                    print(f"  达到收敛条件，误差: {best_error:.12f}")
                    convergence_achieved = True
                    break

                # 检查停滞条件 - 只在启用重启时检查
                if enable_restart and (generation - last_improvement) > self.stagnation_generations:
                    print(f"  检测到停滞，超过{self.stagnation_generations}代无改进，当前误差: {best_error:.12f}")
                    break

                # 生成新一代
                new_population = []

                # 精英保留
                elite = [ind for ind, _ in ranked_population[:self.elite_size]]
                new_population.extend(elite)

                # 生成剩余个体
                while len(new_population) < self.population_size:
                    parent1 = self.selection(ranked_population)
                    parent2 = self.selection(ranked_population)
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutate(child1, generation, self.max_generations)
                    child2 = self.mutate(child2, generation, self.max_generations)
                    new_population.extend([child1, child2])

                population = new_population[:self.population_size]

            # 更新全局最佳
            if best_fitness > best_fitness_overall:
                best_fitness_overall = best_fitness
                best_individual_overall = best_individual
                best_error_overall = best_error

            if enable_restart:
                print(f"第{restart + 1}次重启完成，最终误差: {best_error:.12f}")

            # 如果已经收敛，提前结束
            if convergence_achieved:
                break

        status = "成功收敛" if convergence_achieved else "达到终止条件"
        if enable_restart:
            status += f" (重启{total_restarts_used}次)"
        print(f"优化完成，最终误差: {best_error_overall:.12f} [{status}]")
        return best_individual_overall, best_fitness_history, best_error_history, convergence_achieved


def optimize_all_patterns(nn, patterns):
    """使用高级遗传算法优化所有模式"""
    print("=" * 70)
    print("开始优化所有模式...")
    print("=" * 70)

    # 创建高级遗传算法优化器
    ga = AdvancedGeneticAlgorithmOptimizer(
        nn,
        population_size=500,
        max_generations=1500,
        mutation_rate=0.05,
        crossover_rate=0.92,
        elite_size=20
    )

    optimized_patterns = []
    results = []

    for i, pattern in enumerate(patterns):
        print(f"\n>>> 正在优化第 {i + 1} 个模式:")
        print(f"目标输出: {pattern[1]}")

        # 使用高级遗传算法优化 - 启用重启策略
        best_input, fitness_history, error_history, converged = ga.evolve(
            pattern[1],
            enable_restart=True,  # 启用重启策略
            max_restarts=3
        )

        # 构建完整输入
        full_input = best_input + [0.0001]
        actual_output = nn.update(full_input)
        final_error = nn.calculate_error(full_input, pattern[1])

        # 存储优化后的模式
        optimized_patterns.append([full_input, pattern[1]])

        # 存储结果
        result = {
            'pattern_index': i + 1,
            'optimized_input': full_input,
            'target_output': pattern[1],
            'actual_output': actual_output,
            'error': final_error,
            'converged': converged
        }
        results.append(result)

        print(f"优化后的输入: [{full_input[0]:.8f}, {full_input[1]:.8f}, {full_input[2]:.8f}, {full_input[3]:.6f}]")
        print(
            f"网络实际输出: [{actual_output[0]:.8f}, {actual_output[1]:.8f}, {actual_output[2]:.8f}, {actual_output[3]:.8f}]")
        print(f"最终误差: {final_error:.12f}")
        print(f"收敛状态: {'成功收敛' if converged else '未完全收敛'}")
        print("-" * 60)

    return optimized_patterns, results


def demo():
    # 定义目标输出模式
    target_patterns = [
        [0.044995767, 0.233700393, 0.012892205, 0.184324649],
        [0.065722645, 0.231985544, 0.029807641, 0.183760876],
        [0.086453548, 0.230236930, 0.046720379, 0.183230426],
        [0.107196752, 0.228519549, 0.063617521, 0.182697248],
        [0.127922632, 0.226790954, 0.080473571, 0.182188158],
        [0.148621050, 0.225026893, 0.097295420, 0.181672730],
        [0.169314019, 0.223260278, 0.114058760, 0.181161780],
        [0.046433568, 0.254545430, 0.013385019, 0.201128160],
        [0.067184367, 0.252791930, 0.030299855, 0.200547739],
        [0.087985928, 0.251029676, 0.047218257, 0.199995798],
        [0.108793278, 0.249287852, 0.064129569, 0.199438738],
        [0.129605771, 0.247522019, 0.081016391, 0.198905009],
        [0.150365231, 0.245755440, 0.097837799, 0.198378724],
        [0.171129882, 0.244021059, 0.114623486, 0.197844550],
        [0.047879522, 0.275523325, 0.013861055, 0.217979503],
        [0.068677725, 0.273757855, 0.030794775, 0.217394670],
        [0.089562879, 0.271988523, 0.047714215, 0.216808824],
        [0.110423072, 0.270209068, 0.064639113, 0.216241619],
        [0.131265805, 0.268473624, 0.081529128, 0.215681774],
        [0.152151267, 0.266689022, 0.098381724, 0.215129597],
        [0.172972126, 0.264938690, 0.115188043, 0.214585379],
        [0.049315139, 0.296628868, 0.014321095, 0.234866342],
        [0.070182490, 0.294837639, 0.031272071, 0.234258475],
        [0.091123424, 0.293067725, 0.048213990, 0.233654500],
        [0.112065963, 0.291303375, 0.065145600, 0.233076055],
        [0.132984782, 0.289546924, 0.082052373, 0.232498879],
        [0.153933245, 0.287781527, 0.098925230, 0.231929695],
        [0.174878634, 0.286038339, 0.115753765, 0.231361356],
        [0.050773484, 0.317775419, 0.014793986, 0.251774277],
        [0.071697382, 0.315990380, 0.031745379, 0.251144644],
        [0.092682231, 0.314236961, 0.048700496, 0.250543385],
        [0.113703534, 0.312480515, 0.065648355, 0.249941747],
        [0.134740521, 0.310717262, 0.082574960, 0.249348134],
        [0.155767168, 0.308981668, 0.099469698, 0.248762814],
        [0.176817268, 0.307276974, 0.116317384, 0.248177115],
        [0.052242940, 0.339005414, 0.015246906, 0.268706086],
        [0.073247485, 0.337234658, 0.032210716, 0.268080801],
        [0.094298609, 0.335432169, 0.049183013, 0.267435926],
        [0.115397255, 0.333688641, 0.066150220, 0.266823292],
        [0.136506774, 0.331960929, 0.083101784, 0.266209987],
        [0.157635486, 0.330264316, 0.100016497, 0.265626661],
        [0.178764382, 0.328560080, 0.116890325, 0.265028298],
        [0.053738240, 0.360194981, 0.015693271, 0.285617629],
        [0.074802195, 0.358435348, 0.032670069, 0.284948268],      #44
        [0.095946402, 0.356696459, 0.049664424, 0.284304578],
        [0.117127859, 0.354974021, 0.066652666, 0.283673593],
        [0.138340043, 0.353300307, 0.083627283, 0.283069031],
        [0.159531549, 0.351620062, 0.100562200, 0.282467358],
        [0.180754489, 0.349934246, 0.117481077, 0.281878935],
        [0.052420053, 0.234955867, 0.014192806, 0.182420877],
        [0.071629248, 0.233709187, 0.030337537, 0.181899311],      # 51
        [0.090821271, 0.232448319, 0.046452624, 0.181388028],
        [0.110005976, 0.231164586, 0.062535524, 0.180880647],
        [0.129156942, 0.229854580, 0.078577120, 0.180385489],
        [0.148328421, 0.228535553, 0.094568412, 0.179897674],
        [0.167411159, 0.227191766, 0.110492845, 0.179417698],
        [0.053445012, 0.254287744, 0.014676418, 0.198443266],
        [0.072687333, 0.252997872, 0.030810500, 0.197898056],
        [0.091944412, 0.251682239, 0.046935478, 0.197359459],
        [0.111191779, 0.250348622, 0.063035706, 0.196836189],
        [0.130433041, 0.249014891, 0.079103939, 0.196305389],
        [0.149611883, 0.247676663, 0.095098604, 0.195807341],
        [0.168777861, 0.246322877, 0.111043446, 0.195311379],
        [0.054465276, 0.273802736, 0.015129608, 0.214507968],
        [0.073761107, 0.272467660, 0.031287208, 0.213966932],
        [0.093072478, 0.271129089, 0.047421411, 0.213399529],
        [0.112386441, 0.269779689, 0.063529151, 0.21285469],
        [0.131698888, 0.268415114, 0.079591520, 0.212322506],
        [0.150956098, 0.267058513, 0.095622907, 0.211795484],
        [0.170247159, 0.26570421, 0.111595789, 0.211271991],
        [0.055466455, 0.293480335, 0.015577098, 0.230630279],
        [0.074841096, 0.292095374, 0.031740995, 0.230064797],
        [0.094222632, 0.290732683, 0.047887917, 0.229510493],
        [0.113605589, 0.28936689, 0.064011152, 0.228932521],
        [0.132988278, 0.28800031, 0.080100538, 0.228381919],
        [0.152347973, 0.286630522, 0.096137419, 0.227844856],
        [0.171705737, 0.285260755, 0.112136519, 0.227308456],
        [0.056494161, 0.313282941, 0.016008657, 0.246824622],
        [0.075928219, 0.311882394, 0.032184120, 0.246226419],
        [0.095384835, 0.310491574, 0.048345307, 0.245632377],
        [0.114848982, 0.309123442, 0.064475464, 0.245057191],
        [0.134325909, 0.307736909, 0.080583743, 0.244488885],
        [0.153777089, 0.306371403, 0.096649769, 0.243933134],
        [0.173205778, 0.305009356, 0.112677368, 0.243374075],
        [0.057519495, 0.333151503, 0.016429206, 0.263039625],
        [0.077032397, 0.331728906, 0.032618187, 0.262420032],
        [0.096544009, 0.330332020, 0.048795628, 0.261806519],
        [0.116099068, 0.328978673, 0.064943660, 0.261201574],
        [0.135654899, 0.327582856, 0.081074523, 0.260614051],
        [0.155163791, 0.326262297, 0.097163973, 0.260036600],
        [0.174692476, 0.324904196, 0.113212432, 0.259470834],
        [0.058588110, 0.353147206, 0.016840371, 0.279286510],
        [0.078154827, 0.351698978, 0.033044795, 0.278635055],
        [0.097759778, 0.350291698, 0.049230745, 0.278007203],
        [0.117380379, 0.348927561, 0.065409784, 0.277392235],
        [0.137023682, 0.347562215, 0.081560175, 0.276791031],
        [0.156641933, 0.346212034, 0.097681311, 0.276198680],
        [0.176273548, 0.344915862, 0.113753444, 0.275620463],
        [0.059401081, 0.238102849, 0.014667562, 0.180945078],
        [0.077308729, 0.237350791, 0.030168439, 0.180480851],
        [0.095217519, 0.236588051, 0.045618281, 0.180012339],
        [0.113092309, 0.235796269, 0.061035351, 0.179554562],
        [0.130930981, 0.234979921, 0.076404289, 0.179102039],
        [0.148722871, 0.234127339, 0.091715131, 0.178659381],
        [0.166467359, 0.233255219, 0.106961021, 0.178230592],
        [0.059988921, 0.255847752, 0.015116429, 0.196128619],
        [0.077953529, 0.255041381, 0.030617628, 0.195639541],
        [0.095901371, 0.254220419, 0.046060119, 0.195146951],
        [0.113845651, 0.253373521, 0.061494939, 0.194665431],
        [0.131763039, 0.252517891, 0.076886561, 0.194199381],
        [0.149616281, 0.251613661, 0.092209089, 0.193729271],
        [0.167427089, 0.250705211, 0.107465241, 0.193275649],
        [0.060541271, 0.273843102, 0.015549312, 0.211425554],
        [0.078581102, 0.272978224, 0.031057079, 0.210916758],
        [0.096592216, 0.272096684, 0.046510707, 0.210412115],
        [0.114587729, 0.271206688, 0.061958650, 0.209912001],
        [0.132578221, 0.270299575, 0.077361186, 0.209416786],
        [0.150513251, 0.269373040, 0.092709190, 0.208926826],
        [0.168427260, 0.268438981, 0.107985743, 0.208433523],
        [0.061159573, 0.292069754, 0.015953841, 0.226791277],
        [0.079248803, 0.291109622, 0.031472226, 0.226268879],
        [0.097333273, 0.290181219, 0.046941131, 0.225725508],
        [0.115409804, 0.289251298, 0.062398661, 0.225204058],
        [0.133430558, 0.288311926, 0.077812462, 0.224689536],
        [0.151457156, 0.287365906, 0.093175084, 0.224182416],
        [0.169439929, 0.286417377, 0.108484090, 0.223691510],
        [0.061724639, 0.310387460, 0.016344284, 0.242210553],
        [0.079896740, 0.309421449, 0.031877730, 0.241668640],
        [0.098031486, 0.308471080, 0.047370938, 0.241107079],
        [0.116162612, 0.307519208, 0.062820345, 0.240571985],
        [0.134300375, 0.306554516, 0.078260103, 0.240038293],
        [0.152418045, 0.305570997, 0.093642872, 0.239506375],
        [0.170524065, 0.304605163, 0.108964721, 0.239002718],
        [0.062309101, 0.328900018, 0.016714760, 0.257669630],
        [0.080549425, 0.327916539, 0.032260890, 0.257108469],
        [0.098759879, 0.326920196, 0.047790343, 0.256552697],
        [0.116992719, 0.325920190, 0.063264246, 0.255998284],
        [0.135197361, 0.324936300, 0.078698711, 0.255445057],
        [0.153434359, 0.323973671, 0.094112109, 0.254894271],
        [0.171598386, 0.322998661, 0.109465045, 0.254364011],
        [0.062900286, 0.347531355, 0.017073585, 0.273188854],
        [0.081201668, 0.346502470, 0.032632491, 0.272588678],
        [0.099503353, 0.345494047, 0.048172897, 0.272015519],
        [0.117843401, 0.344491445, 0.063694762, 0.271441831],
        [0.136161953, 0.343501729, 0.079156512, 0.270874601],
        [0.154456003, 0.342543872, 0.094580874, 0.270314111],
        [0.172735170, 0.341553079, 0.109960373, 0.269760623],
    ]

    # 创建初始模式（使用随机输入）
    initial_patterns = []
    for target in target_patterns:
        random_input = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 0.0001]
        initial_patterns.append([random_input, target])

    # 创建神经网络
    n = BPNN(4, 12, 4)

    print("=== 第一步：使用高级遗传算法优化输入矢量 ===")
    # 使用高级遗传算法优化所有模式
    optimized_patterns, ga_results = optimize_all_patterns(n, initial_patterns)

    # 统计遗传算法优化结果
    ga_errors = [result['error'] for result in ga_results]
    converged_count = sum(1 for result in ga_results if result['converged'])

    print(f"\n=== 遗传算法优化结果统计 ===")
    print(f"总模式数量: {len(ga_results)}")
    print(f"成功收敛模式: {converged_count}")
    print(f"收敛率: {converged_count / len(ga_results) * 100:.1f}%")
    print(f"平均误差: {np.mean(ga_errors) * 10000:.6f}")
    print(f"最小误差: {np.min(ga_errors) * 10000:.6f}")
    print(f"最大误差: {np.max(ga_errors) * 10000:.6f}")

    print(f"\n=== 第二步：使用优化后的结果对进行网络的输入训练 ===")
    # 使用优化后的模式进行训练
    n.train(optimized_patterns)

    # 绘制误差图
    n.plot_errors()

    # 显示最终优化结果统计
    print(f"\n=== 最终优化结果统计 ===")
    errors = [err[1] for err in n.vector_errors]
    print(f"平均性能指标: {np.mean(errors):.6f}")
    print(f"最小性能指标: {np.min(errors):.6f}")
    print(f"最大性能指标: {np.max(errors):.6f}")

    # 统计达到全局最优的矢量数量
    optimal_threshold = 0.1  # 根据实际情况调整这个阈值
    optimal_count = sum(1 for err in errors if err <= optimal_threshold)
    print(f"达到全局最优的矢量数量: {optimal_count}/{len(errors)} ({optimal_count / len(errors) * 100:.1f}%)")


if __name__ == '__main__':
    demo()
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f} 秒")