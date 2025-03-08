from flask import Blueprint, request, render_template, jsonify
# from data.preprocess_text import preprocess
# from models import your_model  # 假设你有一个模型加载函数

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

# @main.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     text = data['text']
#     processed_text = preprocess(text)  # 这里使用了你自己写的文本处理代码
#     prediction = your_model.predict(processed_text)  # 假设你有预测函数
#     return jsonify({'prediction': prediction})
