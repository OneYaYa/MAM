#Usage examples:

#python use_api.py --question "这张胸片提示什么疾病？肺炎、肺结核、肺癌还是肺气肿" --file "./samples/test.jpg" 
#python use_api.py --question "该音频提示什么疾病？" --file "./samples/test.mp3" 
#python use_api.py --question "视频是什么康复训练？" --file "./samples/rehub.mp4" 


import argparse
import os
import sys
import shutil
import uuid
import time

# ---------- Robust import path handling ----------
# Allow running from repository root or any working dir.
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

# Try common source-layouts: e.g. model/, pipeline/, Google_Search_API_Wrapper/
for sub in ["model", "pipeline", "Google_Search_API_Wrapper"]:
    p = os.path.join(ROOT, sub)
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)

# ---------- Safe CUDA check & environment hints ----------
def ensure_cuda_or_warn():
    try:
        import torch
        has = torch.cuda.is_available()
        if not has:
            print("[WARN] 未检测到可用的 NVIDIA CUDA。部分上游实现默认 device='cuda'，在 CPU 上将报错或极慢。")
            print("       如需在 CPU 上强制运行，需修改上游类构造器默认 device 参数（此脚本不改动上游源码）。")
        else:
            print(f"[INFO] CUDA 可用: {torch.cuda.get_device_name(0)}")
        return has
    except Exception as e:
        print(f"[WARN] CUDA 检测异常: {e}")
        return False

# ---------- Imports from your uploaded modules ----------
try:
    # Pipeline: modality selection / type classification / roles meeting / final diagnosis / review / memory / web-search
    from pipeline.modality_selection import modality_selection
    from pipeline.type_classification import type_classification
    from pipeline.role_generation import generate_role
    from pipeline.web_search_check import WebSearchCheck
    from pipeline.meeting import roles_meeting
    from pipeline.diagnosis import final_diagnosis  
    from pipeline.review import review_all
    from pipeline.memory import memory
except Exception as e:
    print("[FATAL] 导入 pipeline 相关模块失败，请确认目录结构与相对导入。", e)
    sys.exit(1)

# ---------- Utility ----------
def prepare_datadir_and_copy(file_path: str, target_dir: str = "2406_data_example") -> str:
    """
    一些上游函数（如 type_classification）内部固定使用 data_file='2406_data_example' 并用 os.path.join 读取文件。
    为避免修改上游代码，这里将用户提供的文件复制到该目录下，并返回复制后的相对文件名。
    """
    if not file_path:
        return ""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    dst_name = os.path.basename(file_path)
    dst_path = os.path.join(target_dir, dst_name)
    try:
        shutil.copyfile(file_path, dst_path)
    except Exception as e:
        print(f"[WARN] 复制文件到 {dst_path} 失败：{e}。尝试直接使用原路径，但某些函数可能无法读取。")
        return file_path
    return dst_name  # 注意：上游函数期望传入 '文件名' 而非绝对路径

def pretty_bar(title: str):
    print("\n" + "="*20 + f" {title} " + "="*20 + "\n")

def main():
    ensure_cuda_or_warn()

    parser = argparse.ArgumentParser(description="Multimodal Medical Pipeline - use_api")
    parser.add_argument("--question", type=str, required=True, help="医疗问题/任务指令")
    parser.add_argument("--file", type=str, default="", help="可选的多模态文件路径（图像/音频/视频）。若为空则按文本处理。")
    parser.add_argument("--modality", type=str, default="", choices=["", "text", "image", "audio", "video"],
                        help="显式指定模态；留空则自动判断（依据文件后缀）。")
    parser.add_argument("--rounds", type=int, default=3, help="讨论轮数上限(roles_meeting 内部也有限制)。")
    parser.add_argument("--web_search", action="store_true", help="是否在流程中附加 Web 搜索与摘要。")
    parser.add_argument("--save_memory", action="store_true", help="是否将本次问答写入 ./history/*.json")
    args = parser.parse_args()

    question = args.question.strip()
    fpath = args.file.strip()

    # 0) 自动判断模态（若用户没有指定）
    if args.modality:
        modality = args.modality
    else:
        modality = modality_selection(question, fpath)  # 返回 'image' | 'audio' | 'video' | 'text'

    pretty_bar("Step 0. 输入与模态判断")
    print(f"Question: {question}")
    print(f"Input file: {fpath or '(无)'}")
    print(f"Predicted modality: {modality}")

    # 0) 保持与上游接口兼容：将输入文件复制到 data 目录（如需）
    file_name_for_pipeline = ""
    if modality in ("image", "audio", "video") and fpath:
        file_name_for_pipeline = prepare_datadir_and_copy(fpath, target_dir="2406_data_example")
    else:
        file_name_for_pipeline = ""

    # 1) 问题类型 / 影像器官等粗分类
    pretty_bar("Step 1. 问题/数据类型分类")
    try:
        type_name = type_classification(modality, question, file_name_for_pipeline)
        print(f"[type_classification] => {type_name}")
    except Exception as e:
        print(f"[WARN] type_classification 执行失败：{e}")
        type_name = "general"

    # 2) 自动角色生成（GP 视角拆解：专科/放射技师等）
    pretty_bar("Step 2. 自动生成角色任务清单")  
    try:
        roles_generated = generate_role(type_name, modality, question, file_name_for_pipeline)
        print(roles_generated)
    except Exception as e:
        print(f"[WARN] generate_role 执行失败：{e}")
        roles_generated = "**Specialist Doctor** (General):\n- Provide general assessment\n- Request necessary studies\n"

    # 3) 多角色会议讨论与投票，得到会议纪要
    pretty_bar("Step 3. 多角色会议讨论与投票（会议纪要）")
    history_item = ""  # 可根据需要引入既往史等外部上下文
    try:
        meeting_record = roles_meeting(question, file_name_for_pipeline, modality, type_name, roles_generated, history_item)
    except Exception as e:
        print(f"[ERROR] roles_meeting 执行失败：{e}")
        meeting_record = "Meeting aborted due to runtime error."

    # 4) 终局诊断（将会议纪要注入最终问诊提示）
    pretty_bar("Step 4. 最终诊断生成（融合会议纪要）")
    try:
        final_answer = final_diagnosis(question, file_name_for_pipeline, modality, type_name, meeting_record)
        print(final_answer)
    except Exception as e:
        print(f"[ERROR] final_diagnosis 执行失败：{e}")
        final_answer = "Final diagnosis failed due to runtime error."

    # 5) 诊断结果复核（Yes/No）
    pretty_bar("Step 5. 结果自检（Review）")
    try:
        review = review_all(question, file_name_for_pipeline, modality, type_name, final_answer)
        print(f"[review_all] => {review}")
    except Exception as e:
        print(f"[WARN] review_all 执行失败：{e}")
        review = "no"

    # 6) 可选：Web 搜索检核与摘要
    if args.web_search:
        pretty_bar("Step 6. Web 信息检核与摘要（可选）")
        try:
            search_summary = WebSearchCheck(question, file_name_for_pipeline if file_name_for_pipeline else fpath, modality)
            print(search_summary)
        except Exception as e:
            print(f"[WARN] WebSearchCheck 执行失败：{e}")

    # 7) 可选：写入历史 memory
    if args.save_memory:
        pretty_bar("Step 7. 写入历史记录（./history/*.json）")
        os.makedirs("./history", exist_ok=True)
        try:
            uid = str(uuid.uuid4())[:8]
            memory(uid, question, file_name_for_pipeline or fpath, modality, final_answer)
            print(f"[memory] 写入成功，id={uid}")
        except Exception as e:
            print(f"[WARN] memory 写入失败：{e}")

    pretty_bar("Completed")

if __name__ == "__main__":
    main()
