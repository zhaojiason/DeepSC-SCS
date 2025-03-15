# routes/views.py
import os
import json
import torch
from flask import Flask, render_template, request, json
from performance import DeepSC, interactive_performance, parser, SNR_to_noise, greedy_decode

app = Flask(__name__)

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
    print(f"Initializing model on {device}")
    
    # 初始化模型
    model = DeepSC(
        args.num_layers, num_vocab, num_vocab,
        num_vocab, num_vocab, args.d_model, 
        args.num_heads, args.dff, 0.1
    ).to(device)
    
    # 加载检查点
    checkpoint_path = os.path.join(args.checkpoint_path, 'best_checkpoint.pth')
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
SNR = 40  # 固定SNR值

@app.route('/', methods=['GET', 'POST'])
def index():
    output = ""
    if request.method == 'POST':
        user_input = request.form.get('inputText', '').strip()
        
        try:
            if not user_input:
                raise ValueError("Input cannot be empty")
            
            # 调用推理函数（注意传递device参数）
            result = interactive_performance(
                args, SNR, net, user_input,
                token_to_idx, start_idx, end_idx, pad_idx, device  # 新增设备参数
            )
            output = result
        except Exception as e:
            output = f"Error: {str(e)}"
    
    return render_template('index.html', output_text=output)


