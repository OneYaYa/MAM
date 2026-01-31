import json
import os
import re
from pipeline.modality_selection import modality_selection
from model.language_model import MedicalAssistant
from model.image_model import HuatuoChatbot
from model.video_model import VideoLLaMAChatbot
from model.audio_model import AudioChatbot

def type_classification(modality, ques, file_name):
    data_file='2406_data_example'
    if modality=='image':
        image_paths = [os.path.join(data_file,file_name)]
        bot = HuatuoChatbot()
        query = 'Please answer with a single word: What kind of medical image is this? X-Ray, CT, MRI, Pathology, Biomedical'
        output = bot.chat(images=image_paths, text=query)
        if 'x-ray' in output.lower() or 'ct' in output.lower() or 'mri' in output.lower():
            query_more='Please answer with a single word: What part of the human body does this image show? Brain, bone, abdomen, mediastinum, liver, lung, kidney, soft tissue, pelvis'
            output_more = bot.chat(images=image_paths, text=query_more)
            return output, output_more
        return output
    elif modality=='audio':
        bot = AudioChatbot()
        audio_paths = os.path.join(data_file,file_name)
        query = 'Please answer with a single word: What kind of audio is this? Cardiovascular, Respiratory'
        output = bot.chat(audio=audio_paths,text=query)
        return output
    elif modality=='video':
        video_paths = [os.path.join(data_file, file_name)]
        
        # 暂时使用基于文件名和问题的简单分类，避免Flash Attention问题
        file_name_lower = file_name.lower()
        ques_lower = ques.lower()
        
        # 基于文件名推断视频类型
        if 'rehab' in file_name_lower or 'rehabilitation' in file_name_lower or '康复' in ques_lower:
            return 'Rehabilitation'
        elif 'sport' in file_name_lower or 'exercise' in file_name_lower or '运动' in ques_lower:
            return 'Sports'
        elif 'emergency' in file_name_lower or 'accident' in file_name_lower or '急救' in ques_lower:
            return 'Emergency'
        else:
            # 如果无法从文件名推断，则尝试使用VideoLLaMA2
            try:
                bot = VideoLLaMAChatbot()
                query = 'Please answer with a single word: What kind of video is this? Sports, Rehabilitation, Emergency'
                output = bot.chat(paths=video_paths, text=query, modal_type='video')
                return output
            except Exception as e:
                print(f"[WARN] VideoLLaMA2 分析失败，使用默认分类: {e}")
                # 基于问题关键词的后备分类
                if any(keyword in ques_lower for keyword in ['康复', 'rehabilitation', '训练', 'training', '治疗', 'therapy']):
                    return 'Rehabilitation'
                elif any(keyword in ques_lower for keyword in ['运动', 'sport', '锻炼', 'exercise']):
                    return 'Sports'
                else:
                    return 'Rehabilitation'  # 默认为康复训练
    elif modality=='text':
        question_text = ques
        assistant = MedicalAssistant()
        question = f'''
        System prompt: You are given a question, please select a question type according to the given question.
        
        Input: The question is {question_text}. Which kind of question is this? Anaesthesia, Anatomy, Biochemistry, Dental, ENT, FM, O&G, Medicine, Microbiology, Ophthalmology, Orthopaedics, Pathology, Pediatrics, Pharmacology, Physiology, Psychiatry, Radiology, Skin, PSM, Surgery, Unknown.
        
        Output example: 
        The question type is **Anaesthesia**.
        '''
        response = assistant.generate_response(question)
        match = re.search(r'question type is (\w+)', response)
        if match:
            return match.group(1)
        else:
            return 'general'
            
