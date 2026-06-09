import os
import subprocess
import json
import traceback

def get_video_info(ffmpeg_path, video_path):
    ffprobe_path = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg","ffprobe")
    cmd = [ffprobe_path, '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,r_frame_rate,duration', '-of', 'json', video_path]
    try:
        result = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True, check = True)
        info = json.loads(result.stdout)
        if 'streams' in info and len(info['streams']) > 0:
            stream = info['streams'][0]
            width = int(stream.get('width',0))
            height = int(stream.get('height',0))
            duration = float(stream.get('duration',0))
            fps_parts = stream.get('r_frame_rate', '30/1').split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
            return width, height, fps, duration
    except Exception:
        return None
    return None

def process_single_video(ffmpeg_path, video_path, root_output_dir):
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(root_output_dir, video_filename)
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
    meta = get_video_info(ffmpeg_path, video_path)
    if meta:
        w, h, fps, duration = meta
        print(f"\n[视频]:{os.path.basename(video_path)}")
        print(f"分辨率: {w}x{h} | 帧率: {fps:.2f} | 时长：{duration:.2f}秒")
        
    output_pattern = os.path.join(video_output_dir,"gbt_frame_%04d.jpg")
    
    cmd = [ffmpeg_path, '-y', '-i', video_path, '-vf' , 'fps=1/2', '-vsync', 'vfr', '-q:v', '2', output_pattern]
    
    result = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True)
    if result.returncode == 0:
        count = len([f for f in os.listdir(video_output_dir) if f.startswith("gbt_frame_")])
        print(f"抽帧成功 已存入：{video_output_dir}(共{count}帧)")
    else:
        print(f"抽帧失败 视频{os.path.basename(video_path)}发生错误")
    
def batch_gbt_splitter(ffmpeg_path, input_root,output_root):
    print(f"请输入视频文件夹路径：{input_root}")
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')
    video_list = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(video_extensions):
                full_path = os.path.join(root, file)
                video_list.append(full_path)
                    
    total_video = len(video_list)
    print(f"扫描完毕，共发现{total_video}个视频文件。")
        
    if total_video ==  0:
        print("未发现任何有效视频文件，请检查输入文件夹路径是否正确")
        return
        
    for index, video_path in enumerate(video_list, 1):
        print(f"\n 视频处理进度：[{index}/{total_video}]")
        try:
            process_single_video(ffmpeg_path,video_path, output_root)
        except Exception as e:
            print(f"处理视频{video_path}时遭遇未知错误，已自动跳过。错误日志：{e}")
            traceback.print_exc()
    print("\n" + "="*50 + "\n 自动化抽帧任务已完成！")

if __name__ == "__main__":
    ffmpeg_executable = r"E:\FFmpeg\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
    input_root_dir = r"E:\Vehicle_Data_Anonymization_Verifier\self_test_video"
    output_root_dir = r"E:\Vehicle_Data_Anonymization_Verifier\Output_Frame"

batch_gbt_splitter(ffmpeg_executable, input_root_dir, output_root_dir)