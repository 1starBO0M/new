import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import glob
import pyedflib
from scipy import signal
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from tensorflow.keras.utils import to_categorical 
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input,Conv1D,LSTM, Bidirectional,Dense, Dropout, MaxPooling1D, Flatten, Attention, concatenate,Lambda,AveragePooling1D,Reshape,BatchNormalization
from tensorflow.keras.models import  Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GridSearchCV  
from keras.wrappers.scikit_learn import KerasClassifier  
import gc  

def load_data():

    # 定义睡眠阶段（annotation）到标签（label）的映射关系
    ann2label = {
        "Sleep stage W": 0,  # 清醒期
        "Sleep stage 1": 1,  # 睡眠1期
        "Sleep stage 2": 2,  # 睡眠2期
        "Sleep stage 3": 3, "Sleep stage 4": 3,  # 深睡眠期（合并睡眠3期和睡眠4期，遵循AASM手册）
        "Sleep stage R": 4,  # 快速眼动期
        "Sleep stage ?": 6,  # 未知阶段
        "Movement time": 5  # 运动期间
    }
    
    # 指定数据集的目录路径
    data_dir = "E:\\annaconda/data/sleepedf/sleep-cassette"
    # 指定数据集中要选择的通道名称
    # select_ch = "EEG Fpz-Cz"
    
    # 初始化包含153个列向量的列表X和Y
    X = [np.array([1]) for _ in range(153)]
    Y = [np.array([1]) for _ in range(153)]
    
    # 查找并排序文件
    psg_fnames = glob.glob(os.path.join(data_dir, "*PSG.edf"))  # 获取PSG文件列表
    Hypnogram_fnames = glob.glob(os.path.join(data_dir, "*Hypnogram.edf"))  # 获取Hypnogram文件列表
    psg_fnames.sort()  # 对PSG文件列表进行排序
    Hypnogram_fnames.sort()  # 对Hypnogram文件列表进行排序
    psg_fnames = np.asarray(psg_fnames)  # 将PSG文件列表转换为NumPy数组
    Hypnogram_fnames = np.asarray(Hypnogram_fnames)  # 将Hypnogram文件列表转换为NumPy数组
    
    # 遍历每对PSG文件和Hypnogram文件
    for i in range(len(psg_fnames)):
        print(i)
        # 读取PSG文件和Hypnogram文件
        psg_f = pyedflib.EdfReader(psg_fnames[i])
        Hypnogram_f = pyedflib.EdfReader(Hypnogram_fnames[i])
        
        # 验证PSG文件和Hypnogram文件的开始和结束时间是否一致
        assert psg_f.getStartdatetime() == Hypnogram_f.getStartdatetime()
    
        # print(f"PSG 文件持续时间: {psg_f.getFileDuration()} 秒")
        # 获取eeg通道索引
        channels=psg_f.getSignalLabels()
        channel = "EEG Fpz-Cz"
        if channel in channels:# 如果存在，获取该通道的索引
            channel_index = channels.index(channel)
            int(channel_index)
        else:raise ValueError(f"通道 '{channel}' 在文件中未找到.")# 如果不存在，抛出异常
        epoch_duration=30
        signals = psg_f.readSignal(channel_index)# 读取对应通道信号
        sampling_rate = psg_f.getSampleFrequency(channel_index)#对应通道的采样率
        # 进行数据的一致性检查看看有没有epoch时间对不上的
        n_epochs = psg_f.datarecords_in_file #整个数据文件中的数据记录数
        if psg_f.datarecord_duration == 60:  # 修复SC4362F0-PSG.edf和SC4362FC-Hypnogram.edf的问题
            print(f"文件 {psg_f.file_name} 的数据记录持续时长为60秒")
            n_epochs = n_epochs * 2
        assert len(signals)/3000 == n_epochs, f"signal: {signals.shape} != {n_epochs}"
        signals = signals.reshape(-1,int( epoch_duration*sampling_rate))#整成（n，3000）
        # 读取注释并生成标签
        Hypnogram_onsets,  Hypnogram_durations,  Hypnogram_stages= Hypnogram_f.readAnnotations()
        #定义持续时间验证
        epoch_duration=30
        labels=[]
        duration=0
        #遍历每一个阶段的注释（未划分）
        for a in range(len(Hypnogram_stages)):
            now_duration=int(Hypnogram_durations[a])
            Hypnogram_onsets=None
            duration = now_duration + duration# 更新总持续时间
            # 获取当前睡眠阶段注释对应的标签值
            ann_str = "".join(Hypnogram_stages[a])  # 将当前阶段的注释连接成一个字符串
            label = ann2label[ann_str]  # 获取当前阶段注释对应的标签值
            now_epoch = int(now_duration / epoch_duration)  # 计算当前注释阶段的 epoch 数量
            # 生成标签数组并将其添加到标签列表中
            #创建长度为 now_epoch 始化为 1的数组
            label_epoch = np.ones(now_epoch, dtype=None) * label 
            labels.extend(label_epoch)
        
        # 将所有标签数组水平堆叠成一个单独的数组
        labels = np.hstack(labels)
        # 关闭文件
        psg_f.close()
        Hypnogram_f.close()
        #看看睡眠时长能不能整除30s，不能的话就是有问题
        if duration % epoch_duration != 0:
            print(f"Something wrong: {duration} {epoch_duration}")
        #删除比记录的信号更长的注释 
        
        labels = labels[:len(signals)]
        # 将信号数据转换为 float32 类型，将标签数据转换为 int32 类型
        x = signals.astype(np.float32)
        y = labels.astype(np.int32)
        
        # 只选择睡眠时段，不包括清醒期(W)
        w_edge_mins = 30  # 设置清醒期边界时间（分钟）
        epoch_duration = 30  # 每个epoch的持续时间（秒）
        # 找到所有非清醒期(W)的索引
        nw_idx = np.where(y != ann2label["Sleep stage W"])[0]
        # 计算睡眠时段的起始索引，向前延伸w_edge_mins * 2分钟
        start_idx = nw_idx[0] - (w_edge_mins * 2) * epoch_duration
        # 计算睡眠时段的结束索引，向后延伸w_edge_mins * 2分钟
        end_idx = nw_idx[-1] + (w_edge_mins * 2) * epoch_duration
        start_idx = max(start_idx, 0)    # 确保起始索引不小于0
        end_idx = min(end_idx, len(x)-1)    # 确保结束索引不大于信号序列的长度
        select_idx = np.arange(start_idx, end_idx + 1)    # 生成选择的索引区间
        # 根据索引区间选择信号和标签数据   
        x = x[select_idx]  # 更新信号数据
        y = y[select_idx]  # 更新标签数据
        
        # 去除运动(MOVE)和未知(UNK)阶段
        move_idx = np.where(y == ann2label["Movement time"])[0]  # 找到运动期的索引
        unk_idx = np.where(y == ann2label["Sleep stage ?"])[0]  # 找到未知期的索引
        
        # 如果存在运动期或未知期的索引
        if len(move_idx) > 0 or len(unk_idx) > 0:
            remove_idx = np.union1d(move_idx, unk_idx)  # 合并两个索引为一个去除非睡眠阶段的索引数组
        
            # 从选择的索引中排除去除的索引
            select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
        
            # 根据更新后的索引选择信号和标签数据
            x = x[select_idx]  # 再次更新信号数据，去除不需要的阶段
            y = y[select_idx]  # 再次更新标签数据，去除不需要的阶段
        X[i]=x
        Y[i]=y
    x=None
    y=None
    ann2label=None
    data_dir=None
    Hypnogram_fnames=None
    Hypnogram_durations=None
    labels=None
    label_epoch=None
    nw_idx=None
    psg_f=None
    psg_fnames=None
    select_idx=None
    signals=None
    duration=None
    end_idx=None
    l=None
    gc.collect()
    X= np.vstack(X)
    Y= np.hstack(Y)

    return X,Y


def balance(X, Y):
    # 获取所有类别标签# 重新采样以达到类别平衡 X1,X2,X3,Y1,Y2,Y3 
    gc.collect()
    standard = 12000  # 找到合适的类别样本数量作为标准  
    counts = np.bincount(Y)# 计算每个类别的样本数量
    print(counts)
    print("Standard:", standard)  
    balance_X_list = []
    balance_Y_list = []
    for c in np.unique(Y):  
        idx = np.where(Y == c)[0]  # 找到当前类别的样本索引  
        if len(idx) < standard:# 如果当前类别样本数量小于标准，则复制样本  
            n_repeats = int(standard / len(idx))  # 向下取整  
            tmp_X = np.repeat(X[idx], n_repeats, axis=0)  
            tmp_Y = np.repeat(Y[idx], n_repeats)  
              
            n_remain = standard - len(tmp_Y)  # 如果重复后的样本数量仍小于标准，随机选择样本补足差额  
            if n_remain > 0:  
                additional_idx = np.random.choice(idx, n_remain, replace=True)  
                tmp_X = np.vstack([tmp_X, X[additional_idx]])  
                tmp_Y = np.hstack([tmp_Y, Y[additional_idx]])  
        else:  
            # 如果当前类别样本数量大于或等于标准，随机选择样本以减少到标准数量  
            keep_idx = np.random.choice(idx, standard, replace=False) 
            tmp_X = X[keep_idx]  
            tmp_Y = Y[keep_idx]  
        balance_X_list.append(tmp_X)  # 将处理后的 X 添加到列表中  
        balance_Y_list.append(tmp_Y)  # 将处理后的 Y 添加到列表中  
    # 将列表转换为 NumPy 数组  
    balance_X = np.vstack(balance_X_list)  
    balance_Y = np.concatenate(balance_Y_list)  # 使用 concatenate 而不是 hstack，因为 Y 可能不是二维的  
    tmp_X = None  
    tmp_Y = None 
    X=None
    Y=None
    keep_idx=None
    nw_idx=None
    idx=None
    counts = [len(y) for y in balance_Y_list] # 计算平衡后每个类别的样本数量（基于列表）  
    
    balance_X_list = None  
    balance_Y_list = None
    y=None
    print("Balanced Counts:", counts)  
    print('balance_X.shape', balance_X.shape)  
    print('balance_Y.shape', balance_Y.shape)
    return balance_X, balance_Y



def filters(eeg_data):
    cutoff =49.99    # 滤波器截止频率（Hz）
    order = 4       # 滤波器阶数
    Fs=100
    # 设计低通滤波器
    Wn = cutoff / (0.5 * Fs)
    b, a = signal.butter(order, Wn, btype='lowpass')
    
    # 应用滤波器到脑电数据
    filtered_data = signal.filtfilt(b, a, eeg_data)
    #
    return filtered_data


# 构建DCNN_BiLSTM模型
def model(num_conv_layers=6, filters=16, kernel_size=5, num_lstm_units=64,activation='relu', num_attention_units=32,optimizer='adam' ,input_shape=(3000, 1)):
    input_layer = Input(shape=input_shape)  
    batchnormal= BatchNormalization() (input_layer) 
    # 初始化卷积层 (conv_layer)(input_layer )
    conv_layer= Conv1D(filters=filters, kernel_size=kernel_size, activation=activation)(batchnormal)
    for i in range(num_conv_layers):         
        conv_layer = Conv1D(filters=filters * (2 **i), kernel_size=kernel_size, activation=activation)(conv_layer)   # 添加卷积层（这里只添加一个卷积层）  
        # 在这里，我们交替使用池化层和Dropout层  
        if i % 2 == 0 and i != num_conv_layers - 1:  # 只在偶数迭代（除了最后一次）添加池化层  
            conv_layer = MaxPooling1D(pool_size=4)(conv_layer)  # 减小池化大小以避免过度降采样  
        elif i % 2 == 1:  # 在奇数迭代添加Dropout层  
            conv_layer = Dropout(rate=0.5)(conv_layer)  
    branch = Conv1D(filters= filters, kernel_size= 3, activation=activation)(batchnormal)
    # 根据 num_conv_layers 动态添加卷积层和池化层/Dropout层  
    for i in range(num_conv_layers):  
        # 添加卷积层（这里只添加一个卷积层）  
        branch = Conv1D(filters=filters * (2 **i), kernel_size=3, activation=activation)(branch)  
        # 在这里，我们交替使用池化层和Dropout层  
        if i % 2 == 0 and i != num_conv_layers - 1:  # 只在偶数迭代（除了最后一次）添加池化层  
            branch  = MaxPooling1D(pool_size=2)(branch)  # 减小池化大小以避免过度降采样  
        elif i % 2 == 1:  # 在奇数迭代添加Dropout层 
            branch  = Dropout(rate=0.5)( branch )
    branch = Lambda(lambda x: x[:, :40, :])(branch) 
    ADD=concatenate([conv_layer, branch])
    dropout3 = AveragePooling1D(pool_size=2) (ADD)
    # Bidirectional LSTM  
    lstm = Bidirectional(LSTM(num_lstm_units, return_sequences=False)) ( dropout3)
    attention_layer=Attention()([lstm,lstm])
    dense_layer= Dense(128, activation='relu') (attention_layer)
    # 全连接层，使用softmax激活函数
    flatten_layer = Flatten()(dense_layer)
    dense_layer = Dense(128, activation='relu')(flatten_layer) 
    output_layer = Dense(5, activation='softmax')(dense_layer) 
    # 定义模型  
    model = Model(inputs=input_layer, outputs=output_layer)  
    # 编译模型  
    model.compile(loss='categorical_crossentropy',  optimizer=optimizer, metrics=['accuracy'])    
    model.summary()    
    return model
# model()
def main():
    # tf.debugging.set_log_device_placement(True)
    # X,Y=load_data()
    X=np.load('E:\\睡眠分期\数据集\EEG-Fpz-Cz\X.npy')
    Y=np.load('E:\\睡眠分期\数据集\EEG-Fpz-Cz\Y.npy')
    print('X.shape',X.shape)
    print('Y.shape',Y.shape)

    # 将平衡后的数据划分为训练集和测试集
    balance_X, balance_Y=balance(X, Y)
    # #低通过滤
    for l in range(len(balance_X[:,0])) :
        balance_X[l]  =filters(balance_X[l])
    # 划分训练集和测试集和验证集
    X_train, X_test, Y_train, Y_test = train_test_split(balance_X, balance_Y, test_size=0.2,random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    balance_X=None
    balance_Y=None
    # 转换标签为one-hot编码
    Y_train_one_hot = to_categorical(Y_train, num_classes=5)
    Y_val_one_hot = to_categorical(Y_val, num_classes=5)
    Y_test_one_hot= to_categorical(Y_test, num_classes=5)
    # Y_train =None
    # Y_val = None
    # Y_test=None
    # # 定义参数网格  
    # param_grid = dict(kernel_size=[3,5,7]) 
    gc.collect()  # 手动触发垃圾回收
    # #实例化
    # estimator = KerasClassifier(build_fn=model, epochs=10,batch_size=48, verbose=0)  
    # grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)  
    # grid_result = grid.fit(X_train, Y_train_one_hot) 
    # best_model = grid.best_estimator_ 
    # # 查看最佳参数和评分  
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))  
    #   # 使用测试集评估最佳模型的性能  
    # 添加回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    my_model=model() 
    history =my_model.fit(X_train, Y_train_one_hot, validation_data=(X_val, Y_val_one_hot), callbacks=callbacks,epochs=200,batch_size=32, use_multiprocessing=True)
    # 绘制训练和验证的损失
    plt.plot(history.history['loss'], label='Training loss: {:.2f}'.format(min(history.history['loss'])))
    plt.plot(history.history['val_loss'], label='Validation loss: {:.2f}'.format(min(history.history['val_loss'])))
    plt.title('Loss Over Training Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 1.0) 
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')  # Add a legend to the plot
    plt.show()
    # 绘制训练和验证的准确率
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy: {:.2f}'.format(max(history.history['accuracy'])))
    plt.plot(history.history['val_accuracy'], label='Validation accuracy: {:.2f}'.format(max(history.history['val_accuracy'])))
    plt.title('Accuracy Over Training Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0.4, 1) 
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    scores = my_model.evaluate(X_test, Y_test_one_hot, verbose=1)   
    print("测试集上的准确率: %.2f%%" % (scores[1]*100))
    # 打印训练损失和准确率  
    print("Training Loss:")  
    print(history.history['loss'])  
    print("Training Accuracy:")  
    print(history.history['accuracy'])  
    # 打印验证损失和准确率（如果有的话）  
    print("Validation Loss:")  
    print(history.history['val_loss'])  
    print("Validation Accuracy:")  
    print(history.history['val_accuracy'])
    # 评估模型
    # 使用模型进行预测
    y_pred_proba = my_model.predict(X_test)  # 获取概率预测
    y_pred = y_pred_proba.argmax(axis=-1)  # 将概率转换为最可能的类别
        
    accuracy = accuracy_score(Y_test, y_pred)# 计算整体准确率
    error_rate = 1 - accuracy# 计算错误率
    conf_matrix = confusion_matrix(Y_test, y_pred)# 打印混淆矩阵
    class_report = classification_report(Y_test, y_pred, zero_division=0)# 打印分类报告，包括每个类别的精确度、召回率和F1分数
    weighted_f1 = f1_score(Y_test, y_pred, average='weighted')# 计算加权平均F1分数
    macro_f1 = f1_score(Y_test, y_pred, average='macro') # 计算宏平均F1分数
    
    print(f'Macro F1 Score: {macro_f1:.2f}')
    print(f'Weighted F1 Score: {weighted_f1:.2f}')
    print('Classification Report:\n', class_report)
    print('Confusion Matrix:\n', conf_matrix)
    print(f'Error Rate: {error_rate:.2f}')
    print(f'Accuracy: {accuracy:.2f}')
    # # 加载最佳模型
    # best_model = load_model('best_model.h5')
    # # 定义回调函数
    # checkpoint = ModelCheckpoint(
    #     filepath='best_model.h5',  # 文件路径
    #     monitor='val_loss',  # 监控的量度
    #     save_best_only=True,  # 只保存最佳模型
    #     mode='min',  # 因为我们希望最小化损失
    #     verbose=1  # 详细模式
    # )
    # 保存模型参数
    my_model.save_weights('m4_weights_acc80%.h5')
    my_model.save('entire_m3_acc80.h5')
    current_directory = os.getcwd()
    print("当前工作目录：", current_directory)
    # from tensorflow.keras.models import load_model
    # # 加载模型参数
    # model.load_weights('model_weights.h5')
    # # 加载整个模型（包括结构和参数）
    # model = load_model('entire_model.h5')
    # for l in range(len(X_val[:,0])) :
    #     X_val[l]  =filter(X_val[l])
    # # 3. 绘制EEG时间序列图    
    # Fs = 100  # 假设采样频率为100Hz
    # t = np.arange(0, len(X_val[9]) / Fs, 1 / Fs)
    # plt.figure(figsize=(10, 4))
    # plt.plot(t, X_val[9], color='blue')
    # plt.xlabel('Time (s)')
    # plt.ylabel('EEG Amplitude')
    # plt.title('EEG Time Series of Wake1')
    # plt.grid(True)
    # plt.show()
    result_path='E:\睡眠分期\算法'
    model_struct_img = os.path.join(result_path, "m3_acc77%.png")
    plot_model(my_model,to_file=model_struct_img,
                show_shapes=True,
                show_layer_names=True)
# 主函数执行的部分保持不变
if __name__ == "__main__":
    main()