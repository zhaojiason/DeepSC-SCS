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
def initialize_model():
    # 解析参数
    args = parser.parse_args()
    
    # 加载词汇表
    vocab_filename = 'vocab.json'
    args.vocab_file = os.path.join('data', args.vocab_file, vocab_filename)
    with open(args.vocab_file, 'r') as f:
        vocab = json.load(f)
    token_to_idx = vocab['token_to_idx']
    idx_to_token = {v: k for k, v in token_to_idx.items()}
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    
    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = DeepSC(
        args.num_layers, num_vocab, num_vocab,
        num_vocab, num_vocab, args.d_model, 
        args.num_heads, args.dff, 0.1
    ).to(device)
    
    # 加载检查点
    checkpoint_path = os.path.join(f'checkpoints/{config.channel_type}', 'best_checkpoint.pth')
    print("Load model from: ", checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # 设备感知加载
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 修复可能的键名不匹配
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    
    return args, model, token_to_idx, idx_to_token, start_idx, end_idx, pad_idx, device

# ============== 在应用启动时初始化 ==============
args, net, token_to_idx, idx_to_token, start_idx, end_idx, pad_idx, device = initialize_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    output = ""
    similarity = 0.0  # 初始化默认值
    if request.method == 'POST':
        user_input = request.form.get('inputText', '').strip()
        
        try:
            if not user_input:
                raise ValueError("Input cannot be empty")
            
            # 使用全局配置参数
            output, similarity = interactive_performance(
                args, 
                config.snr,        # 使用动态SNR
                net, 
                user_input,
                token_to_idx, 
                start_idx, 
                end_idx, 
                pad_idx, 
                device,
                channel=config.channel_type  # 添加信道类型参数
            )
            print("Channel: ", config.channel_type)
            print("SNR value: ", config.snr)
        except Exception as e:
            output = f"Error: {str(e)}"
            # 可以选择重置相似度或保持默认值
            similarity = 0.0
    
    return render_template('index.html', output_text=output, similarity=round(similarity, 6))

# ============== 新增配置更新路由 ==============
@app.route('/update_config', methods=['POST'])
def update_config():
    try:
        data = request.get_json()
        # 参数验证
        required_fields = ['channelType', 'snr']
        if not all(field in data for field in required_fields):
            return jsonify({'status': 'error', 'message': 'Missing parameters'}), 400

        # 验证信道类型
        valid_channels = ['AWGN', 'Rayleigh', 'Rician']
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
