import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BioGptTokenizer, BioGptForCausalLM

class MedicalAssistant:
    def __init__(self, model_name="microsoft/BioGPT", device=None): # 更换为 BioGPT 模型
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        

        #print("------------------checkpoint5-------------------")
        
        # 使用 BioGPT 专用的 tokenizer 和 model，强制使用 safetensors 格式
        try:
            #print("尝试加载 BioGPT 专用 tokenizer...")
            self.tokenizer = BioGptTokenizer.from_pretrained(model_name)
            
            # 修复 attention mask 警告：设置 pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                #print("已设置 pad_token 以避免 attention mask 警告")
            
            #print("尝试加载 BioGPT 专用模型（使用 safetensors）...")
            # 强制使用 safetensors 格式来避免 PyTorch 安全限制
            self.model = BioGptForCausalLM.from_pretrained(
                model_name,
                use_safetensors=True,  # 强制使用 safetensors
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)


        except Exception as e:
            print(f"使用 BioGPT 专用类失败，回退到通用 Auto 类: {e}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # 修复 attention mask 警告：设置 pad_token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    print("已设置 Auto tokenizer 的 pad_token")
                
                print("Auto tokenizer 加载成功")
                # 也在 Auto 类中强制使用 safetensors
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    use_safetensors=True,  # 强制使用 safetensors
                    trust_remote_code=True
                ).to(self.device)
                print("Auto model 加载成功（safetensors）")
            except Exception as e2:
                print(f"Auto类也失败了: {e2}")
                raise e2
        self.sys_message = ''' 
        You are a medical assistant specializing in clinical analysis. Provide clear, concise medical assessments in plain text. Focus on practical diagnostic considerations and recommendations. Avoid mathematical formulas, XML tags, or complex formatting. Use simple, professional language suitable for healthcare professionals.
        '''

    def format_prompt(self, question, sys_message=None):
        # 改进的医疗提示格式，避免产生XML内容
        if not sys_message:
            sys_content = self.sys_message.strip()
        else:
            sys_content = sys_message.strip()
        
        # 简化提示格式，避免复杂输出
        prompt = f"{sys_content}\n\nQuestion: {question}\nProvide a clear medical assessment:"
        return prompt

    def generate_response(self, question, sys_message=None, max_new_tokens=512):
        prompt = self.format_prompt(question, sys_message)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        
        # 确保 attention_mask 被正确设置
        if 'attention_mask' not in inputs:
            attention_mask = (inputs['input_ids'] != self.tokenizer.pad_token_id).long()
            inputs['attention_mask'] = attention_mask
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                use_cache=True,
                do_sample=True,
                temperature=0.8,  # BioGPT 适合稍高的温度
                top_p=0.9,
                repetition_penalty=1.1,  # 减少重复
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        
        answer = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        # 移除输入提示部分
        answer = answer[len(prompt):].strip()
        return answer

if __name__ == "__main__":
    print("初始化 BioGPT 医疗助手...")
    assistant = MedicalAssistant()
    
    question = '''
    Patient presents with chest pain, shortness of breath, and elevated troponin levels. 
    What are the possible diagnoses and recommended diagnostic workup?
    '''
    response = assistant.generate_response(question)
    print(response)
