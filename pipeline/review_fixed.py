from model.language_model import MedicalAssistant
from model.image_model import HuatuoChatbot
from model.video_model import VideoLLaMAChatbot
from model.audio_model import AudioChatbot

def get_review_prompt(ques,record):
    prompt = f'''
    Input: You're a medical assistant. Please check whether the answer to this question is reasonable, if it is, please answer "yes", if not, please answer "no"

    Question: {ques}.

    Answer: {record}
    '''

    return prompt

def review_all(question, file_name, modality_type, type_name, result_diagnosis):
    query = get_review_prompt(question, result_diagnosis)
    if modality_type=='image':
        bot = HuatuoChatbot()
        output = bot.chat(images=file_name, text=query)
    elif modality_type=='audio':
        bot = AudioChatbot()
        output = bot.chat(audio=file_name,text=query)
    elif modality_type=='video':
        # 暂时使用文本模型处理视频分析，避免Flash Attention问题
        try:
            bot = VideoLLaMAChatbot()
            output = bot.chat(paths=file_name, text=query, modal_type='video')
        except Exception as e:
            print(f"[WARN] VideoLLaMA2 review失败，使用文本模型: {e}")
            # 使用文本模型提供基于诊断结果的复核
            assistant = MedicalAssistant()
            fallback_query = f'''
            作为医学助手，请检查以下关于康复训练视频的诊断是否合理。如果合理请回答"yes"，如果不合理请回答"no"。

            问题: {question}
            
            诊断结果: {result_diagnosis}
            
            注意：由于视频分析模块暂时不可用，请基于医学知识判断诊断的合理性。
            '''
            output = assistant.generate_response(fallback_query)
    elif modality_type=='text':
        assistant = MedicalAssistant()
        output = assistant.generate_response(query)
    return output