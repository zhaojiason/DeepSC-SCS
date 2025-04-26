# routes/views.py
import os
import json
import torch
from flask import Flask, render_template, request, json, jsonify
from performance import DeepSC, interactive_performance, parser

app = Flask(__name__)
# ============== 全局配置参数 ==============
class ModelConfig:
    def __init__(self):
        self.channel_type = 'AWGN'  # 默认信道类型
        self.snr = 20  # 默认SNR值

# 初始化全局配置对象
config = ModelConfig()
# ============== 模型初始化函数 ==============
def initialize_model(channel_type):
    """动态初始化模型，根据传入的信道类型加载对应检查点"""
    # 解析基础参数（这些参数在应用启动后通常不会改变）
    args = parser.parse_args()
    
    # 加载词汇表
    with open(args.vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # 构建映射表
    token_to_idx = vocab
    idx_to_token = {v: k for k, v in token_to_idx.items()}

    # 提取特殊标记
    pad_idx = token_to_idx.get('[PAD]', None)
    start_idx = token_to_idx.get('[CLS]', None)
    end_idx = token_to_idx.get('[SEP]', None)
    num_vocab = len(token_to_idx)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型架构
    model = DeepSC(
        args.num_layers, num_vocab, num_vocab,
        num_vocab, num_vocab, args.d_model,
        args.num_heads, args.dff, 0.1
    ).to(device)
    
    # 动态构建检查点路径
    checkpoint_path = os.path.join(f'checkpoints/{channel_type}_data1', 'best_checkpoint.pth')
    print(f"Loading model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # 加载模型参数
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    
    return args, model, token_to_idx, idx_to_token, start_idx, end_idx, pad_idx, device

@app.route('/', methods=['GET', 'POST'])
def index():
    # GET请求直接返回空页面
    if request.method == 'GET':
        return render_template('index.html', 
                            input_text='',
                            output_text='',
                            similarity=None)

    # POST请求处理逻辑
    output = ""
    similarity = 0.0
    user_input = request.form.get('inputText', '').strip()
    
    try:
        if not user_input:
            raise ValueError("Input cannot be empty")
        
        # 动态获取当前配置
        current_channel = config.channel_type
        current_snr = config.snr
        
        # 初始化模型
        args, net, token_to_idx, idx_to_token, start_idx, end_idx, pad_idx, device = initialize_model(current_channel)
        
        # 处理请求
        output, similarity = interactive_performance(
            args,
            current_snr,
            net,
            user_input,
            token_to_idx,
            start_idx, 
            end_idx,
            pad_idx,
            device,
            channel=current_channel
        )
        
        print(f"Current Channel: {current_channel}")
        print(f"SNR: {current_snr}")
    except Exception as e:
        output = f"Error: {str(e)}"
        similarity = 0.0

    # 保留用户输入并返回结果
    return render_template('index.html',
                        input_text=user_input,  # 关键：回传用户输入
                        output_text=output,
                        similarity=round(similarity, 6) if similarity else None)

@app.route('/update_config', methods=['POST'])
def update_config():
    try:
        data = request.get_json()
        # 参数验证
        required_fields = ['channelType', 'snr']
        if not all(field in data for field in required_fields):
            return jsonify({'status': 'error', 'message': 'Missing parameters'}), 400

        # 验证信道类型
        valid_channels = ['AWGN', 'Rayleigh', 'Rician','Suzuki','Nakagami']
        if data['channelType'] not in valid_channels:
            return jsonify({'status': 'error', 'message': 'Invalid channel type'}), 400

        # 验证SNR
        try:
            snr = int(data['snr'])
        except ValueError:
            return jsonify({'status': 'error', 'message': 'SNR must be integer'}), 400
            
        if not (0 <= snr <= 20):
            return jsonify({'status': 'error', 'message': 'SNR out of range'}), 400

        config.channel_type = data['channelType']
        config.snr = snr

        return jsonify({
            'status': 'success',
            'new_config': {
                'channelType': config.channel_type,
                'snr': config.snr
            }
        })
        
    except Exception as e:
        app.logger.error(f'Config update error: {str(e)}')
        return jsonify({'status': 'error', 'message': 'Internal error'}), 500

@app.route('/get_config', methods=['GET'])
def get_current_config():
    return jsonify({
        'channelType': config.channel_type,
        'snr': config.snr
    })
